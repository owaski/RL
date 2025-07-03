ROOT=/lustre/fs11/portfolios/convai/users/souyang

CONTAINER_PATH=$ROOT/images/nemo_rl.sqsh
CODE_DIR=$ROOT/code
CKPTS_DIR=$ROOT/ckpts
DATA_DIR=$ROOT/data
HF_CACHE_DIR=$ROOT/.cache/huggingface

NUM_ACTOR_NODES=1  # Total nodes requested (head is colocated on ray-worker-0)

MOUNTS="/lustre/fs11:/lustre/fs11,${CODE_DIR}:/code,${CKPTS_DIR}:/ckpts,${DATA_DIR}:/data" \
CONTAINER=${CONTAINER_PATH} \
HF_DATASETS_CACHE=$HF_CACHE_DIR \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=convai_convaird_nemo-speech \
    --job-name=grpo-dev-infinisst \
    --partition=interactive \
    --time=2:0:0 \
    --gres=gpu:8 \
    ray.sub

export NRL_VLLM_USE_V1=0
export PYTHONPATH=/code/RL:$PYTHONPATHuv 
uv python examples/run_grpo_infinisst.py