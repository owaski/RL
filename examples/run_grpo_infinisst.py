# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import itertools
import os
import pprint
import random
from typing import Any, Iterator

import torch
import numpy as np
import pandas as pd

from omegaconf import OmegaConf
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer

from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env
from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.games.infinisst import InfiniSSTEnv
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir

OmegaConf.register_new_resolver("mul", lambda a, b: a * b)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run GRPO training with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    args, overrides = parser.parse_known_args()
    return args, overrides

CODE2LANG = {
    'zh': 'Chinese',
    'en': 'English',
    'ja': 'Japanese',
    'ko': 'Korean',
    'fr': 'French',
    'de': 'German',
    'es': 'Spanish',
}

INSTRUCTION = "Translate the following speech from {} to {}."

class IterableInfiniSSTDataset(IterableDataset):
    """An IterableDataset that generates sliding puzzle data indefinitely."""

    def __init__(
        self, tokenizer, data_file, shuffle, seed, src_lang, tgt_lang, task_name, length
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.df = pd.read_parquet(data_file)
        self.shuffle = shuffle
        self.seed = seed
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.task_name = task_name
        self.length = length

    def __iter__(self) -> Iterator[DatumSpec]:
        print("Starting IterableInfiniSSTDataset (indefinite generation).")
        # Use itertools.count for an infinite index generator
        df = self.df.sample(frac=1, random_state=self.seed) if self.shuffle else self.df
        for i in itertools.count():
            row = df.iloc[i % len(df)]
            data = row.to_dict()

            features = np.load(data['audio_npy_path'], mmap_mode='r')[data['audio_npy_row']]
            first_step_features = features[:data['chunk_frame_size']]
            
            instruction = INSTRUCTION.format(CODE2LANG[self.src_lang], CODE2LANG[self.tgt_lang])

            message_log = [
                {
                    "role": "system",
                    "content": instruction,
                },
                {
                    "role": "user",
                    "content": "<|video_pad|>" * data['chunk_frame_size'],
                    "features": first_step_features,
                }
            ]
            tokenized_prompt = self.tokenizer.apply_chat_template(
                message_log,
                return_tensors="pt",
                add_special_tokens=False,
                add_generation_prompt=True,
            )[0]
            message_log[1]['token_ids'] = tokenized_prompt

            datum: DatumSpec = {
                'message_log': message_log,
                'length': len(tokenized_prompt),
                'extra_env_info': {
                    'features': features,
                    'step': 0,
                    'max_steps': features.shape[0] // data['chunk_frame_size'],
                    'chunk_frame_size': data['chunk_frame_size'],
                    'src_segments': data['src_segments'],
                    'tgt_segments': data['tgt_segments'],
                },
                'loss_multiplier': 1.0,
                'idx': i,
                'task_name': self.task_name,
            }
            yield datum

    def __len__(self):
        return self.length


def setup_infinisst_data(
    tokenizer: AutoTokenizer,
    env_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    task_name: str,
    length: int,
    val_length: int,
) -> tuple[IterableDataset, IterableDataset | None, dict, dict]:
    """Sets up the iterable data generator and env map for the sliding puzzle task."""
    print("Setting up InfiniSST iterable data and environment...")
    env_config = env_cfg[task_name]

    print(f"Instantiating environment for task '{task_name}'...")
    env = InfiniSSTEnv.options(
        num_gpus=env_config["cfg"]["num_gpus"],
        runtime_env={
            "py_executable": get_actor_python_env(
                "nemo_rl.environments.games.infinisst.InfiniSSTEnv"
            ),
            "env_vars": dict(os.environ),  # Pass thru all user environment variables
        }
    ).remote(cfg=dict(env_config["cfg"]))
    task_to_env = {task_name: env}
    print(f"Environment '{task_name}' created.")

    print("Creating InfiniSST dataset...")
    training_dataset = IterableInfiniSSTDataset(
        tokenizer=tokenizer,
        data_file=data_cfg["train_data_file"],
        shuffle=data_cfg["train_data_shuffle"],
        seed=data_cfg["seed"],
        src_lang=data_cfg["src_lang"],
        tgt_lang=data_cfg["tgt_lang"],
        task_name=task_name,
        length=length,
    )
    print("InfiniSST dataset created.")

    validation_dataset = IterableInfiniSSTDataset(
        tokenizer=tokenizer,
        data_file=data_cfg["val_data_file"],
        shuffle=data_cfg["val_data_shuffle"],
        seed=data_cfg["seed"],
        src_lang=data_cfg["src_lang"],
        tgt_lang=data_cfg["tgt_lang"],
        task_name=task_name,
        length=val_length,
    )
    val_task_to_env = task_to_env

    return training_dataset, validation_dataset, task_to_env, val_task_to_env


def main():
    """Main entry point."""
    # Parse arguments
    args, overrides = parse_args()

    print("This is my nemo-rl repo.")

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "grpo_infinisst.yaml"
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")

    # Print config
    print("Final config:")
    pprint.pprint(config)

    # Get the next experiment directory with incremented ID
    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"📊 Using log directory: {config['logger']['log_dir']}")
    if config["checkpointing"]["enabled"]:
        print(
            f"📊 Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}"
        )

    init_ray()

    # setup tokenizer
    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"], tokenizer
    )

    # setup data & env map
    ds_length = (
        config["grpo"]["num_prompts_per_step"]
        * config["grpo"]["num_generations_per_prompt"]
        * config["grpo"]["max_num_steps"]
    )
    dataset, val_dataset, task_to_env, val_task_to_env = setup_infinisst_data(
        tokenizer=tokenizer,
        env_cfg=config["env"],
        data_cfg=config["data"],
        task_name="infinisst",
        length=ds_length,
        val_length=config["grpo"]["max_val_samples"],
    )

    (
        policy,
        policy_generation,
        cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    ) = setup(config, tokenizer, dataset, val_dataset)

    grpo_train(
        policy,
        policy_generation,
        dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        task_to_env,
        val_task_to_env,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    )


if __name__ == "__main__":
    main()
