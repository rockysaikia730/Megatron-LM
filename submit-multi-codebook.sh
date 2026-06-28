#!/bin/bash
#
# Two modes (toggle with MODE below):
#
#   MODE=test   Tiny toy model + mock data on 2 nodes (8 GPUs).
#                Validates TP × PP × CP × SP wiring + multi-codebook plumbing
#                without depending on any preprocessed dataset. ~5 minutes.
#
#   MODE=fleurs  Full Llama3-8B + real FLEURS-en_us HCodec audio on 1 node
#                (4 GPUs). The configuration that produced our PoC numbers.
#                
#

#SBATCH --account=infra01
#SBATCH --time=01:00:00
#SBATCH --partition=debug
#SBATCH --environment=megatronedf
#SBATCH --job-name=rvq
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM/logs/slurm/training/%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM/logs/slurm/training/%x-%j.err
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --no-requeue

################ MODE selection ################
# Override at submit time with:  MODE=fleurs sbatch submit-multi-codebook.sh
MODE="${MODE:-test}"

# Validate
if [[ "$MODE" != "test" && "$MODE" != "fleurs" ]]; then
    echo "ERROR: MODE must be 'test' or 'fleurs'; got '$MODE'" >&2
    exit 1
fi

echo "===================================================================="
echo "MODE:       $MODE"
echo "START TIME: $(date)"
echo "JOB ID:     ${SLURM_JOB_ID:-<none>}"
echo "NODES:      ${SLURM_NNODES:-<unset>}"
echo "===================================================================="


################ Per-mode configuration ################
if [[ "$MODE" == "test" ]]; then
    # 2 nodes × 4 GPUs = 8 GPUs total
    NNODES=2
    GPUS_PER_NODE=4
    NTASKS_PER_NODE=4

    # Parallelism: TP=2 × PP=1 × CP=2 × DP=2 = 8 GPUs
    TP_SIZE=2
    PP_SIZE=1
    CP_SIZE=2

    # Tiny model
    NUM_LAYERS=4
    HIDDEN_SIZE=256
    FFN_HIDDEN_SIZE=512
    NUM_ATTN_HEADS=4
    NUM_QUERY_GROUPS=2
    SEQ_LEN=128

    # Mock-data tokenizer / vocab
    TOKENIZER_TYPE="NullTokenizer"
    TEXT_VOCAB_SIZE=1024

    # Multi-codebook: scaled down so test runs fast
    NUM_AUDIO_CB=4
    AUDIO_VOCAB=128
    AUDIO_PAD_ID=127

    # Training budget
    MBS=2
    GBS=4
    TRAINING_STEPS=20
    LR_WARMUP=5

    EXP_NAME="mcb-test-${NNODES}n-tp${TP_SIZE}-pp${PP_SIZE}-cp${CP_SIZE}"

elif [[ "$MODE" == "fleurs" ]]; then
    # 1 node × 4 GPUs = 4 GPUs total
    NNODES=1
    GPUS_PER_NODE=4
    NTASKS_PER_NODE=4

    # Parallelism: TP=4 × PP=1 × CP=1 × DP=1 = 4 GPUs (the PoC config)
    TP_SIZE=4
    PP_SIZE=1
    CP_SIZE=1

    # Full Llama3-8B
    NUM_LAYERS=32
    HIDDEN_SIZE=4096
    FFN_HIDDEN_SIZE=14336
    NUM_ATTN_HEADS=32
    NUM_QUERY_GROUPS=8
    SEQ_LEN=2048

    # Real swissai tokenizer
    TOKENIZER_TYPE="HuggingFaceTokenizer"
    TOKENIZER_MODEL="alehc/swissai-tokenizer"
    # The preprocessor wrote these IDs as the audio span markers; must match.
    AUDIO_START_ID=131072
    AUDIO_END_ID=131073

    # Multi-codebook: full HCodec sizing
    NUM_AUDIO_CB=4
    AUDIO_VOCAB=1024
    AUDIO_PAD_ID=1023

    # Training budget
    MBS=1
    GBS=4
    TRAINING_STEPS=500
    LR_WARMUP=20

    # FLEURS .bin/.idx prefix written by tools/audio/preprocess_fleurs_hcodec.py
    DATA_PATH_PREFIX="/iopsstor/scratch/cscs/$USER/datasets/fleurs_en_us_hcodec/train"

    EXP_NAME="llama3-8b-fleurs-${NNODES}n-tp${TP_SIZE}-pp${PP_SIZE}-cp${CP_SIZE}"
fi

AUDIO_LOSS_WEIGHT=1.0
CHECKPOINT_STEPS=$(( TRAINING_STEPS / 2 ))

#### Debugging ####
LOG_NCCL=false
###################


################ Paths ################
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM
HF_HOME_DIR=/iopsstor/scratch/cscs/$USER/hf_cache

PROJECT_NAME="Megatron-Clariden-MultiCodebook"
PROJECT_DIR="$MEGATRON_LM_DIR/logs/Meg-Runs/$PROJECT_NAME"
EXP_DIR="$PROJECT_DIR/$EXP_NAME"
CKPT_DIR="$EXP_DIR/checkpoints"
TRIGGER_DIR="$EXP_DIR/triggers"
DEBUG_DIR="$EXP_DIR/debug/${SLURM_JOB_ID:-local}"
LOGGING_DIR="$EXP_DIR/logging"
TENSORBOARD_DIR="$LOGGING_DIR/tensorboard"
COMPUTE_ENVIRONMENT_FILE="$DEBUG_DIR/compute_environment.txt"
GPU_MEM_LOG="$DEBUG_DIR/memory_logging.txt"

mkdir -p "$CKPT_DIR" "$TRIGGER_DIR" "$DEBUG_DIR" "$LOGGING_DIR"

# Wipe stale exit/save triggers from prior runs in the same EXP_DIR. Without
# this, training exits after one iter because the .touch() check at the end
# of the previous run left logs/triggers/exit behind.
rm -f "$TRIGGER_DIR/exit" "$TRIGGER_DIR/save"


################ Distributed env ################
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# Required for TP and CP; if unset, Megatron asserts and exits.
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# HF cache on /iopsstor (compute nodes are offline; preload on a login node
# beforehand: HF_HOME=$HF_HOME_DIR python -c "from transformers import
# AutoTokenizer; AutoTokenizer.from_pretrained('alehc/swissai-tokenizer')").
export HF_HOME="$HF_HOME_DIR"
export TRANSFORMERS_OFFLINE=1

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29501

ulimit -c 0


################ Megatron args ################
TRANSFORMER_ENGINE_ARGS=(
    --transformer-impl transformer_engine
)

NETWORK_SIZE_ARGS=(
    --num-layers $NUM_LAYERS
    --hidden-size $HIDDEN_SIZE
    --ffn-hidden-size $FFN_HIDDEN_SIZE
    --num-attention-heads $NUM_ATTN_HEADS
    --group-query-attention
    --num-query-groups $NUM_QUERY_GROUPS
    --max-position-embeddings $SEQ_LEN
    --position-embedding-type rope
    --rotary-base 500000
    --make-vocab-size-divisible-by 128
    --normalization RMSNorm
    --swiglu
    --untie-embeddings-and-output-weights
)

MULTI_CODEBOOK_ARGS=(
    --enable-multi-codebook-heads
    --num-audio-codebooks $NUM_AUDIO_CB
    --audio-codebook-size $AUDIO_VOCAB
    --audio-pad-token-id $AUDIO_PAD_ID
    --audio-loss-weight $AUDIO_LOSS_WEIGHT
)

LOGGING_ARGS=(
    --log-throughput
    --log-progress
    --tensorboard-dir "$TENSORBOARD_DIR"
    --no-log-loss-scale-to-tensorboard
    --log-memory-to-tensorboard
)

REGULARIZATION_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --weight-decay 0.1
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-beta2 0.95
)

TRAINING_ARGS=(
    --micro-batch-size $MBS
    --global-batch-size $GBS
    --train-iters $TRAINING_STEPS
    --log-interval 1
    --eval-iters 0
    --eval-interval 1000
    --disable-bias-linear
    --optimizer adam
    --dataloader-type single
)

INITIALIZATION_ARGS=(
    --seed 42
    --init-method-std 0.02
)

LEARNING_RATE_ARGS=(
    --lr 0.0003
    --min-lr 0.00003
    --lr-decay-style cosine
    --lr-warmup-iters $LR_WARMUP
)

CHECKPOINTING_ARGS=(
    --save "$CKPT_DIR"
    --save-interval $CHECKPOINT_STEPS
    --ckpt-format torch_dist
    --load "$CKPT_DIR"
    --trigger-path "$TRIGGER_DIR"
)

MIXED_PRECISION_ARGS=(
    --bf16
)

DISTRIBUTED_ARGS=(
    --tensor-model-parallel-size $TP_SIZE
    --pipeline-model-parallel-size $PP_SIZE
    --context-parallel-size $CP_SIZE
    --sequence-parallel
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
    # TE generates causal masks internally; the dataloader version causes a
    # TP-broadcast count mismatch with AudioTextGPTDataset (which omits the
    # attention_mask key). Always off.
    --no-create-attention-mask-in-dataloader
)

# Mode-specific tokenizer + data args
if [[ "$MODE" == "test" ]]; then
    TOKENIZER_ARGS=(
        --tokenizer-type NullTokenizer
        --vocab-size $TEXT_VOCAB_SIZE
    )
    DATA_ARGS=(
        --mock-data
        --split 100,0,0
        --seq-length $SEQ_LEN
        --num-workers 1
        --num-dataset-builder-threads 1
    )
elif [[ "$MODE" == "fleurs" ]]; then
    TOKENIZER_ARGS=(
        --tokenizer-type $TOKENIZER_TYPE
        --tokenizer-model "$TOKENIZER_MODEL"
    )
    DATA_ARGS=(
        --data-path 1.0 "$DATA_PATH_PREFIX"
        --multi-codebook-data
        --audio-start-id $AUDIO_START_ID
        --audio-end-id $AUDIO_END_ID
        --split 100,0,0
        --seq-length $SEQ_LEN
        --num-workers 1
        --num-dataset-builder-threads 1
    )
fi


################ Compose the command ################
cd "$MEGATRON_LM_DIR"
export PYTHONPATH="$MEGATRON_LM_DIR:$PYTHONPATH"

CMD_PREFIX="numactl --membind=0-3"

TRAINING_CMD="python3 $MEGATRON_LM_DIR/pretrain_gpt.py \
    ${TRANSFORMER_ENGINE_ARGS[@]} \
    ${NETWORK_SIZE_ARGS[@]} \
    ${MULTI_CODEBOOK_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${REGULARIZATION_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${INITIALIZATION_ARGS[@]} \
    ${LEARNING_RATE_ARGS[@]} \
    ${CHECKPOINTING_ARGS[@]} \
    ${MIXED_PRECISION_ARGS[@]} \
    ${DISTRIBUTED_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    ${DATA_ARGS[@]}"

if [ "$LOG_NCCL" = true ]; then
    CMD_PREFIX="NCCL_DEBUG=INFO NCCL_DEBUG_FILE=$DEBUG_DIR/nccl-info-\$SLURMD_NODENAME-rank-\$SLURM_PROCID.txt $CMD_PREFIX"
fi


################ Debug snapshot ################
cp "$0" "$DEBUG_DIR/"
{
    echo "$(date)"
    printf '=%.0s' {1..100}; echo
    echo "MODE: $MODE"
    echo "EXP_NAME: $EXP_NAME"
    echo "CMD: $CMD_PREFIX $TRAINING_CMD"
    printf '=%.0s' {1..100}; echo
    echo "NODES: $(scontrol show hostnames $SLURM_JOB_NODELIST)"
    printf '=%.0s' {1..100}; echo
    echo "Megatron path: $MEGATRON_LM_DIR ($(git -C $MEGATRON_LM_DIR rev-parse --verify HEAD 2>/dev/null || echo unknown))"
} > "$COMPUTE_ENVIRONMENT_FILE"


################ Launch ################
# Per-node GPU memory snapshot (quick "did the container come up?" check)
srun -lu --mpi=pmix --network=disable_rdzv_get \
    --nodes=$NNODES --ntasks-per-node=1 --gpus-per-node=$GPUS_PER_NODE \
    bash -c 'echo $(hostname) $(nvidia-smi | grep -o "|\s*[0-9]*MiB" | head -8)' \
    > "$GPU_MEM_LOG" 2>&1 || true

# The actual training launch.
#
# --mpi=pmix:  required by the NGC container's MPI; without it, container
#              startup fails with PMIX errors on some Alps allocations.
# --network=disable_rdzv_get: skips a Slingshot rendezvous step that hangs
#              on certain nodes. Safe to leave on.
srun -lu --mpi=pmix --network=disable_rdzv_get \
    --nodes=$NNODES --ntasks-per-node=$NTASKS_PER_NODE \
    --gpus-per-node=$GPUS_PER_NODE --cpus-per-task=$SLURM_CPUS_PER_TASK \
    bash -c "
        export LD_LIBRARY_PATH=\$(echo \$LD_LIBRARY_PATH | tr ':' '\n' | grep -v compat | paste -sd ':' -)
        if [ -d /usr/local/cuda/compat ]; then
            mv /usr/local/cuda/compat /usr/local/cuda/compat_disabled 2>/dev/null || true
        fi
        export RANK=\$SLURM_PROCID
        export LOCAL_RANK=\$SLURM_LOCALID
        export WORLD_SIZE=\$SLURM_NPROCS
        $CMD_PREFIX $TRAINING_CMD
    "

EXIT_CODE=$?
echo "===================================================================="
echo "END TIME: $(date)"
echo "EXIT CODE: $EXIT_CODE"
echo "===================================================================="
exit $EXIT_CODE