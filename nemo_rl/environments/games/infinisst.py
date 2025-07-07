import os
import copy
import random
from typing import Any, Optional, TypedDict

import ray
import torch
from transformers import AutoTokenizer
import time

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

SPACY_MODELS = {
    "en": "en_core_web_sm",
    "ru": "ru_core_news_sm",
    "de": "de_core_news_sm",
    "zh": "zh_core_web_sm",
    "ja": "ja_ginza_electra",
    "es": "es_core_news_sm"
}

@ray.remote
class InfiniSSTScorer:
    def __init__(self, cfg: InfiniSSTConfig):
        import spacy
        from comet import download_model, load_from_checkpoint

        self.cfg = cfg
        spacy.cli.download(SPACY_MODELS[cfg["tgt_lang"]])
        self.segmenter = spacy.load(SPACY_MODELS[cfg["tgt_lang"]])

        model_path = download_model(cfg["scoring_model_type"], saving_directory=cfg["scoring_model_path"])
        self.scoring_model = load_from_checkpoint(model_path)
        self.batch_size = cfg["batch_size"]
        self.granularity = cfg["granularity"]

    def predict(self, data: list[dict[str, str]]) -> list[float]:
        return self.scoring_model.predict(data, batch_size=self.batch_size, gpus=1).scores

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
        # TODO: delay compute reward until generate is finished
        # TODO: segment the hypothesis into sentences given reference translation sentences

        # TODO: compute latency contribution of each token in the hypothesis

        # TODO: evaluate translation quality of the given granularity

        comet_data = []
        for message_log, metadata in zip(message_log_batch, metadata_batch):
            translation = ''.join([msg["content"] for msg in message_log if msg["role"] == "assistant"])
            reference = ''.join(metadata["tgt_segments"])
            source = ''.join(metadata["src_segments"])
            comet_data.append({
                "src": source,
                "mt": translation,
                "ref": reference,
            })
        n_worker = len(self.workers)
        comet_data_per_worker = [comet_data[i::n_worker] for i in range(n_worker)]
        results = ray.get([self.workers[i].predict.remote(comet_data_per_worker[i]) for i in range(n_worker)])
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