import copy
import random
from typing import Any, Optional, TypedDict

import ray
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)

class InfiniSSTConfig(TypedDict):
    scoring_model_path: str
    scoring_model_type: str
    batch_size: int
    granularity: str

class InfiniSSTMetadata(TypedDict):
    features: torch.Tensor
    step: int
    max_steps: int
    src_segments: list[str]
    tgt_segments: list[str]
    chunk_frame_size: int

@ray.remote
class InfiniSSTEnv(EnvironmentInterface):
    """InfiniSST environment (Ray Actor)."""

    def __init__(self, cfg: Optional[InfiniSSTConfig] = None):
        from comet import download_model, load_from_checkpoint
        self.cfg = cfg
        model_path = download_model(cfg["scoring_model_type"], saving_directory=cfg["scoring_model_path"])
        self.scoring_model = load_from_checkpoint(model_path)
        self.batch_size = cfg["batch_size"]
        self.granularity = cfg["granularity"]

    def compute_reward(self, message_log: LLMMessageLogType, metadata: InfiniSSTMetadata) -> float:
        # TODO: delay compute reward until generate is finished
        # TODO: segment the hypothesis into sentences given reference translation sentences

        # TODO: compute latency contribution of each token in the hypothesis

        # TODO: evaluate translation quality of the given granularity

        return 0.0

    def step(
        self, message_log_batch: list[LLMMessageLogType], metadata_batch: list[InfiniSSTMetadata]
    ) -> EnvironmentReturn:
        observations = []
        rewards = []
        terminateds = []
        all_stop_strings = []
        all_next_metadata = []

        for message_log, metadata in zip(message_log_batch, metadata_batch):
            step = metadata["step"]
            chunk_frame_size = metadata["chunk_frame_size"]
            if step == metadata["max_steps"] - 1:
                reward = self.compute_reward(message_log, metadata)
                rewards.append(reward)
                terminateds.append(True)
                observations.append({
                    "role": "user",
                    "content": "<|video_pad|>" * chunk_frame_size,
                    "features": None,
                })
            else:
                rewards.append(0)
                terminateds.append(False)
                observations.append({
                    "role": "user",
                    "content": "<|video_pad|>" * chunk_frame_size,
                    "features": metadata["features"][: (step + 1) * chunk_frame_size],
                })
            all_stop_strings.append(None)
            metadata["step"] += 1
            all_next_metadata.append(metadata)

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