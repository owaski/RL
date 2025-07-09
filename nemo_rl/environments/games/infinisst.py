import os
import copy
import random
import re
from typing import Any, Optional, TypedDict

import ray
import torch
from transformers import AutoTokenizer
import time
from tqdm import tqdm

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)
from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES, RayVirtualCluster

class InfiniSSTConfig(TypedDict):
    scoring_model_path: str
    scoring_model_type: str
    batch_size: int
    granularity: str
    max_turns: int
    src_lang: str
    tgt_lang: str

class InfiniSSTMetadata(TypedDict):
    features: torch.Tensor
    step: int
    max_steps: int
    src_segments: list[str]
    tgt_segments: list[str]
    chunk_frame_size: int

SENT_SPLITTERS = {
    "en": '.,!?',
    "ru": '.,!?',
    "zh": '。，！？',
}

class LCME:
    def __init__(self, cfg: InfiniSSTConfig):
        from nemo_rl.environments.games.lcme.wmtAlign import load_alternative_model

        self.cfg = cfg
        self.tokenizer, self.model = load_alternative_model('cuda', 'BAAI/bge-m3')

    def segment(self, data: list[dict[str, str]]) -> list[str]:
        from nemo_rl.environments.games.lcme.wmtAlign import generate_overlap_and_embedding, run_vecalign_explore
        src_tgt_alignmentss = []
        features_to_overlap_emb = {}
        for idx, instance in enumerate(tqdm(data, desc="Running VecAlign")):
            tgt_sentences = instance["tgt_sents"]
            src_sentences = instance["src_sents"]
            ref_sentences = instance["ref_sents"]
            features_id, doc_id = instance["id"].split('_')

            if len(tgt_sentences) == 0:
                src_tgt_alignmentss.append([
                    ([idx], [])
                    for idx in range(len(src_sentences))
                ])
                continue
            
            if features_id not in features_to_overlap_emb:
                src_overlap, src_embed = generate_overlap_and_embedding("\n".join(src_sentences), self.model, self.tokenizer, 10)
                features_to_overlap_emb[features_id] = (src_overlap, src_embed)
            else:
                src_overlap, src_embed = features_to_overlap_emb[features_id]
            
            tgt_overlap, tgt_embed = generate_overlap_and_embedding("\n".join(tgt_sentences), self.model, self.tokenizer, 10)

            src_tgt_alignments = run_vecalign_explore(
                "\n".join(src_sentences), "\n".join(tgt_sentences),
                src_overlap, tgt_overlap, src_embed, tgt_embed,
                doc_id, 10
            )

            src_tgt_alignmentss.append(src_tgt_alignments)
        return src_tgt_alignmentss

@ray.remote
class InfiniSSTScorer:
    def __init__(self, cfg: InfiniSSTConfig):
        self.cfg = cfg
        self.sent_splitter = SENT_SPLITTERS[cfg["tgt_lang"]]
        self.segmenter = LCME(cfg)

        from comet import download_model, load_from_checkpoint
        model_path = download_model(cfg["scoring_model_type"], saving_directory=cfg["scoring_model_path"])
        self.scoring_model = load_from_checkpoint(model_path)
        self.worst_score = 0 if 'comet' in cfg["scoring_model_type"].lower() else -25
        self.batch_size = cfg["batch_size"]
        self.granularity = cfg["granularity"]

    def predict(self, data: list[dict[str, str]]) -> list[float]:
        for instance in data:
            tgt_text = instance["tgt_text"]

            tgt_sentences = []
            paragraphs = tgt_text.split('\n')
            for paragraph in paragraphs:
                if paragraph.strip():
                    sentences = []
                    current_sentence = ""
                    for char in paragraph:
                        current_sentence += char
                        if char in self.sent_splitter:
                            if current_sentence.strip():
                                sentences.append(current_sentence.strip())
                            current_sentence = ""
                    if current_sentence.strip():
                        sentences.append(current_sentence.strip())
                    tgt_sentences.extend(sentences)
            
            instance["tgt_sents"] = tgt_sentences

        src_tgt_alignmentss = self.segmenter.segment(data)

        src_sep = '' if self.cfg["src_lang"] in ['zh', 'ja'] else ' '
        tgt_sep = '' if self.cfg["tgt_lang"] in ['zh', 'ja'] else ' '

        rewards = [[] for _ in range(len(data))]
        instance2data = []
        scorer_data = []
        for idx, src_tgt_alignments in enumerate(src_tgt_alignmentss):
            src_sentences = data[idx]["src_sents"]
            ref_sentences = data[idx]["ref_sents"]
            tgt_sentences = data[idx]["tgt_sents"]

            for src_indices, tgt_indices in src_tgt_alignments:
                if len(src_indices) == 0 or len(tgt_indices) == 0:
                    rewards[idx].append(self.worst_score)
                    continue
                src_sentence = src_sep.join([src_sentences[i] for i in src_indices])
                ref_sentence = tgt_sep.join([ref_sentences[i] for i in src_indices])
                tgt_sentence = tgt_sep.join([tgt_sentences[i] for i in tgt_indices])

                scorer_data.append({
                    "src": src_sentence,
                    "ref": ref_sentence,
                    "mt": tgt_sentence,
                })
                instance2data.append(idx)

        scores = self.scoring_model.predict(scorer_data, batch_size=self.batch_size, gpus=1).scores
        for i, idx in enumerate(instance2data):
            rewards[idx].append(scores[i])
        
        mean_rewards = []
        for reward_list in rewards:
            mean_rewards.append(sum(reward_list) / len(reward_list))
        return mean_rewards

@ray.remote
class InfiniSSTEnv(EnvironmentInterface):
    """InfiniSST environment (Ray Actor)."""

    def __init__(self, cfg: Optional[InfiniSSTConfig] = None):
        self.cfg = cfg
        self.virtual_cluster = RayVirtualCluster(
            bundle_ct_per_node_list=[cfg["num_gpus"]],
            use_gpus=True,
            name="infinisst_vc",
        )
        placement_groups = self.virtual_cluster.get_placement_groups()
        
        self.workers = []
        for i in range(cfg["num_gpus"]):
            pg_index = i % len(placement_groups)
            pg = placement_groups[pg_index]
            worker = InfiniSSTScorer.options(
                num_gpus=1,
                runtime_env={
                    "py_executable": get_actor_python_env(
                        "nemo_rl.environments.games.infinisst.InfiniSSTEnv"
                    ),
                },
                scheduling_strategy=ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy(
                    placement_group=pg
                )
            ).remote(cfg)
            self.workers.append(worker)

        self.tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
        self.max_turns = cfg["max_turns"]

    def compute_reward(self, message_log_batch: list[LLMMessageLogType], metadata_batch: list[InfiniSSTMetadata]) -> float:
        scorer_data = []
        for idx, (message_log, metadata) in enumerate(zip(message_log_batch, metadata_batch)):
            translation = ''.join([msg["content"] for msg in message_log if msg["role"] == "assistant"])
            features_id = str(abs(hash(f"{message_log[0]['features'][0]}-{message_log[0]['features'][1]}")))
            scorer_data.append({
                "id": f"{features_id}_{idx}",
                "src_sents": metadata["src_segments"],
                "ref_sents": metadata["tgt_segments"],
                "tgt_text": translation,
            })
        n_worker = len(self.workers)
        scorer_data_per_worker = [scorer_data[i::n_worker] for i in range(n_worker)]
        results = ray.get([self.workers[i].predict.remote(scorer_data_per_worker[i]) for i in range(n_worker)])
        scores = []
        for i in range(len(message_log_batch)):
            scores.append(results[i % n_worker][i // n_worker])
        return scores

    def step(
        self, message_log_batch: list[LLMMessageLogType], metadata_batch: list[InfiniSSTMetadata]
    ) -> EnvironmentReturn:
        start_time = time.time()
        observations = []
        rewards = []
        terminateds = []
        all_stop_strings = []
        all_next_metadata = []

        for metadata in metadata_batch:
            chunk_frame_size = metadata["chunk_frame_size"]
            content = "<|video_pad|>" * chunk_frame_size
            content = self.tokenizer.decode(
                self.tokenizer.apply_chat_template( 
                    [{"role": "user", "content": content}],
                    add_generation_prompt=True,
                    add_special_tokens=False,
                )[20:], # remove system prompt from qwen2.5
            )
            observations.append({"role": "user", "content": content})

            all_stop_strings.append(None)
            metadata["step"] += 1
            all_next_metadata.append(metadata)        
            
        if metadata_batch[0]['step'] == self.max_turns - 1:
            rewards = self.compute_reward(message_log_batch, metadata_batch)
            terminateds = [True] * len(message_log_batch)
        else:
            rewards = [0] * len(message_log_batch)
            terminateds = [False] * len(message_log_batch)

        end_time = time.time()
        elapsed = end_time - start_time
        print(f"InfiniSSTEnv.step took {elapsed:.4f} seconds")

        return EnvironmentReturn(
            observations=observations,
            metadata=all_next_metadata,
            next_stop_strings=all_stop_strings,
            rewards=torch.tensor(rewards, dtype=torch.float32),
            terminateds=torch.tensor(terminateds, dtype=torch.bool),
        )

    def shutdown(self):
        pass

    def global_post_process_and_metrics(self, batch: BatchedDataDict[Any]) -> tuple[BatchedDataDict[Any], dict[str, float | int]]:
        pass