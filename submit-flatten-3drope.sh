#!/bin/bash
#
# Launch flatten + 3D-RoPE (time x depth x stream) RVQ-audio training on Clariden.
#
# This is the --audio-pattern flatten counterpart of submit-multi-codebook.sh:
# every (time, codebook, stream) cell is its own sequence position, audio tokens
# share a single UNION vocabulary with text (one embedding, one head, stock CE),
# and positions carry a genuine 3D rotary embedding via --position-embedding-type
# mrope. No parallel audio heads, no summed embeddings, no delay <audio_pad>.
#
# Two modes (toggle with MODE below):
#
#   MODE=smoke   Tiny model + real FLEURS .bin on 1 node (4 GPUs), TP=2 x CP=2.
#                Exercises the full flatten path INCLUDING the CP position-id
#                handling (the double-slice trap) on real audio. ~few minutes.
#
#   MODE=fleurs  Full Llama3-8B + real FLEURS-en_us HCodec audio on 1 node
#                (4 GPUs), TP=4. Produces the held-out NLL for the comparison
#                against the delay baseline.
#
# Both modes use the SAME audio/vocab config because they read the SAME .bin
# (HCodec V_a=1024); only the transformer size / parallelism / iters differ.
#
#SBATCH --account=infra01
#SBATCH --time=01:00:00
#SBATCH --partition=debug
#SBATCH --environment=megatronedf
#SBATCH --job-name=rvq-flat
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM/logs/slurm/training/%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM/logs/slurm/training/%x-%j.err
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --no-requeue

################ MODE selection ################
# Override at submit time:  MODE=fleurs sbatch submit-flatten-3drope.sh
MODE="${MODE:-smoke}"

if [[ "$MODE" != "smoke" && "$MODE" != "fleurs" ]]; then
    echo "ERROR: MODE must be 'smoke' or 'fleurs'; got '$MODE'" >&2
    exit 1
fi

echo "===================================================================="
echo "MODE:       $MODE  (flatten + 3D-RoPE)"
echo "START TIME: $(date)"
echo "JOB ID:     ${SLURM_JOB_ID:-<none>}"
echo "NODES:      ${SLURM_NNODES:-<unset>}"
echo "===================================================================="


################ Audio / vocab config (shared; matches the .bin) ################
# Markers the preprocessor wrote into the .bin. MUST match your data.
AUDIO_START_ID=131072
AUDIO_END_ID=131073
# HCodec sizing.
NUM_AUDIO_CB=4         # K
AUDIO_VOCAB=1024       # V_a per codebook
NUM_STREAMS=2          # acoustic + semantic
STREAM_ORDER="semantic_first"
# Union vocab: text+markers live below AUDIO_VOCAB_BASE, audio ids above it.
#   audio_id = AUDIO_VOCAB_BASE + stream*(K*V_a) + k*V_a + value
AUDIO_VOCAB_BASE=131074                                  # just above the markers
AUDIO_BLOCK=$(( NUM_STREAMS * NUM_AUDIO_CB * AUDIO_VOCAB ))   # 8192
UNION_TOP=$(( AUDIO_VOCAB_BASE + AUDIO_BLOCK ))               # 139266


################ Per-mode configuration ################
if [[ "$MODE" == "smoke" ]]; then
    NNODES=1
    GPUS_PER_NODE=4
    NTASKS_PER_NODE=4

    # TP=2 x PP=1 x CP=2 x DP=1 = 4 GPUs (validates the CP position-id path).
    TP_SIZE=2
    PP_SIZE=1
    CP_SIZE=2

    # Tiny model. head_dim = HIDDEN/HEADS = 256/4 = 64 -> 32 rotary pairs.
    NUM_LAYERS=4
    HIDDEN_SIZE=256
    FFN_HIDDEN_SIZE=512
    NUM_ATTN_HEADS=4
    NUM_QUERY_GROUPS=2
    SEQ_LEN=1024
    MROPE_SECTION="24 4 4"          # time depth stream; sums to head_dim/2 = 32

    MBS=1
    GBS=4
    TRAINING_STEPS=10
    LR_WARMUP=2

    EXP_NAME="flat-smoke-${NNODES}n-tp${TP_SIZE}-pp${PP_SIZE}-cp${CP_SIZE}"

elif [[ "$MODE" == "fleurs" ]]; then
    NNODES=1
    GPUS_PER_NODE=4
    NTASKS_PER_NODE=4

    # TP=4 x PP=1 x CP=1 x DP=1 = 4 GPUs (the headline config).
    TP_SIZE=4
    PP_SIZE=1
    CP_SIZE=1

    # Full Llama3-8B. head_dim = 4096/32 = 128 -> 64 rotary pairs.
    NUM_LAYERS=32
    HIDDEN_SIZE=4096
    FFN_HIDDEN_SIZE=14336
    NUM_ATTN_HEADS=32
    NUM_QUERY_GROUPS=8
    # Flatten ~4x's the audio length, so give it room (HCodec ~50Hz x K x 2).
    SEQ_LEN=8192
    MROPE_SECTION="56 4 4"          # time depth stream; sums to head_dim/2 = 64

    MBS=1
    GBS=4
    TRAINING_STEPS=500
    LR_WARMUP=20

    EXP_NAME="llama3-8b-flat-fleurs-${NNODES}n-tp${TP_SIZE}-pp${PP_SIZE}-cp${CP_SIZE}"
fi

# Override the step count at submit time:  TRAIN_ITERS=2 MODE=smoke sbatch ...
TRAINING_STEPS="${TRAIN_ITERS:-$TRAINING_STEPS}"

# FLEURS .bin/.idx prefix from tools/audio/preprocess_fleurs_hcodec.py (both modes).
DATA_PATH_PREFIX="/iopsstor/scratch/cscs/$USER/datasets/fleurs_en_us_hcodec/train"

# Padded union vocab: round UNION_TOP up to a multiple of (128 * TP) so the
# output layer shards cleanly across TP. Passed explicitly so the model embedding
# /head cover the audio ids (the HF tokenizer alone would size it to 131072).
VOCAB_DIV=$(( 128 * TP_SIZE ))
PADDED_VOCAB=$(( ( (UNION_TOP + VOCAB_DIV - 1) / VOCAB_DIV ) * VOCAB_DIV ))

CHECKPOINT_STEPS=$(( TRAINING_STEPS / 2 ))

echo "vocab: text/markers < $AUDIO_VOCAB_BASE ; audio block $AUDIO_BLOCK ; "
echo "       union top $UNION_TOP -> padded $PADDED_VOCAB (div $VOCAB_DIV)"

#### Debugging ####
LOG_NCCL=false
###################


################ Paths ################
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM
HF_HOME_DIR=/iopsstor/scratch/cscs/$USER/hf_cache

PROJECT_NAME="Megatron-Clariden-Flatten3DRoPE"
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

# Wipe stale exit/save triggers so we don't exit after one iter.
rm -f "$TRIGGER_DIR/exit" "$TRIGGER_DIR/save"


################ Distributed env ################
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1     # required for TP/CP, else Megatron asserts
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# HF cache on /iopsstor (compute nodes are offline; preload on a login node:
#   HF_HOME=$HF_HOME_DIR python -c "from transformers import AutoTokenizer; \
#   AutoTokenizer.from_pretrained('alehc/swissai-tokenizer')").
export HF_HOME="$HF_HOME_DIR"
export TRANSFORMERS_OFFLINE=1

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29502

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
    # 3D rotary: text positions degenerate to 1D; audio = (time, depth, stream).
    --position-embedding-type mrope
    --mrope-section $MROPE_SECTION
    --rotary-base 500000
    --make-vocab-size-divisible-by 128
    --normalization RMSNorm
    --swiglu
    --untie-embeddings-and-output-weights
)

# Flatten-mode audio args. NOTE: NO --enable-multi-codebook-heads (single head),
# NO --audio-pad-token-id (no delay gaps).
FLATTEN_ARGS=(
    --num-audio-codebooks $NUM_AUDIO_CB
    --audio-codebook-size $AUDIO_VOCAB
    --audio-num-streams $NUM_STREAMS
    --audio-stream-order $STREAM_ORDER
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

# CHECKPOINTING_ARGS, DATA_ARGS, and EVAL_CONTROL_ARGS are set in the train-vs-
# eval toggle block below (they differ between a training run and EVAL=true).

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
    # mrope/flatten get_batch returns attention_mask=None; TE builds the causal
    # mask internally. Keep the dataloader from adding one (TP broadcast mismatch).
    --no-create-attention-mask-in-dataloader
)

TOKENIZER_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "alehc/swissai-tokenizer"
    # Force the union vocab so the embedding/head cover the audio ids. Without
    # this the HF tokenizer would size the model to 131072 and audio ids would
    # be out of range.
    --padded-vocab-size $PADDED_VOCAB
)

################ Train vs held-out eval toggle ################
# EVAL=true  -> load the trained checkpoint and run held-out eval ONLY (no
#               training) over the dev .bin, reporting audio_token_loss + the
#               per-stream/codebook NLL via the Step-1 modality buckets.
# Preprocess the dev set first, e.g.:
#   python tools/audio/preprocess_fleurs_hcodec.py --split validation \
#     --output-prefix /iopsstor/scratch/cscs/$USER/datasets/fleurs_en_us_hcodec/dev ...
# Then:  EVAL=true MODE=fleurs sbatch submit-flatten-3drope.sh
# Set EVAL_ITERS so EVAL_ITERS*GBS covers the dev docs (= ceil(num_dev_docs/GBS))
# for an unbiased pass; AudioTextGPTDataset cycles docs if you over/under-shoot.
EVAL="${EVAL:-false}"
VALID_DATA_PATH_PREFIX="${VALID_PREFIX:-/iopsstor/scratch/cscs/$USER/datasets/fleurs_en_us_hcodec/dev}"
EVAL_ITERS="${EVAL_ITERS:-100}"

AUDIO_DATA_ARGS=(
    --multi-codebook-data
    --audio-pattern flatten
    --audio-start-id $AUDIO_START_ID
    --audio-end-id $AUDIO_END_ID
    --audio-vocab-base $AUDIO_VOCAB_BASE
    --seq-length $SEQ_LEN
    --num-workers 1
    --num-dataset-builder-threads 1
)

if [[ "$EVAL" == "true" ]]; then
    echo "EVAL mode: held-out eval of $CKPT_DIR on $VALID_DATA_PATH_PREFIX ($EVAL_ITERS iters)"
    # --skip-train skips the training loop; the final do_valid eval still runs.
    # AudioTextGPTDataset ignores --split, so the dev set must be a SEPARATE
    # --valid-data-path (not a split of train) to be genuinely held out.
    DATA_ARGS=(
        --train-data-path 1.0 "$DATA_PATH_PREFIX"
        --valid-data-path 1.0 "$VALID_DATA_PATH_PREFIX"
        "${AUDIO_DATA_ARGS[@]}"
    )
    EVAL_CONTROL_ARGS=( --skip-train --eval-iters $EVAL_ITERS --eval-interval 1000 )
    CHECKPOINTING_ARGS=( --load "$CKPT_DIR" --ckpt-format torch_dist )   # load only, no --save
else
    DATA_ARGS=(
        --data-path 1.0 "$DATA_PATH_PREFIX"
        --split 100,0,0
        "${AUDIO_DATA_ARGS[@]}"
    )
    EVAL_CONTROL_ARGS=( --eval-iters 0 --eval-interval 1000 )
    CHECKPOINTING_ARGS=(
        --save "$CKPT_DIR"
        --save-interval $CHECKPOINT_STEPS
        --ckpt-format torch_dist
        --load "$CKPT_DIR"
        --trigger-path "$TRIGGER_DIR"
    )
fi


################ Compose the command ################
cd "$MEGATRON_LM_DIR"
export PYTHONPATH="$MEGATRON_LM_DIR:$PYTHONPATH"

CMD_PREFIX="numactl --membind=0-3"

TRAINING_CMD="python3 $MEGATRON_LM_DIR/pretrain_gpt.py \
    ${TRANSFORMER_ENGINE_ARGS[@]} \
    ${NETWORK_SIZE_ARGS[@]} \
    ${FLATTEN_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${REGULARIZATION_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${EVAL_CONTROL_ARGS[@]} \
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
    echo "MODE: $MODE (flatten + 3D-RoPE)"
    echo "EXP_NAME: $EXP_NAME"
    echo "PADDED_VOCAB: $PADDED_VOCAB  AUDIO_VOCAB_BASE: $AUDIO_VOCAB_BASE"
    echo "MROPE_SECTION: $MROPE_SECTION"
    echo "CMD: $CMD_PREFIX $TRAINING_CMD"
    printf '=%.0s' {1..100}; echo
    echo "NODES: $(scontrol show hostnames $SLURM_JOB_NODELIST)"
    printf '=%.0s' {1..100}; echo
    echo "Megatron path: $MEGATRON_LM_DIR ($(git -C $MEGATRON_LM_DIR rev-parse --verify HEAD 2>/dev/null || echo unknown))"
} > "$COMPUTE_ENVIRONMENT_FILE"


################ Launch ################
# Per-node GPU memory snapshot (quick "did the container come up?" check).
srun -lu --mpi=pmix --network=disable_rdzv_get \
    --nodes=$NNODES --ntasks-per-node=1 --gpus-per-node=$GPUS_PER_NODE \
    bash -c 'echo $(hostname) $(nvidia-smi | grep -o "|\s*[0-9]*MiB" | head -8)' \
    > "$GPU_MEM_LOG" 2>&1 || true

# The actual training launch.
#   --mpi=pmix                  required by the NGC container's MPI on Alps.
#   --network=disable_rdzv_get  skips a Slingshot rendezvous step that can hang.
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
echo ""
echo "Held-out NLL (the comparison metric): this script trains with"
echo "--split 100,0,0 (AudioTextGPTDataset serves whole documents and builds its"
echo "own permutation, so the train/val split is not honoured). For a clean"
echo "held-out NLL, preprocess the FLEURS *dev* split to its own prefix and run"
echo "an eval-only job (--data-path <dev_prefix> --train-iters 0 --eval-iters N,"
echo "or load this run's checkpoint). The per-token val loss IS the audio NLL in"
echo "nats; multiply by tokens/sec / ln(2) for bits/sec."
exit $EXIT_CODE
