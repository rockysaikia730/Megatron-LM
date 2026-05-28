#!/bin/bash

#SBATCH --account=a-a06
#SBATCH --time=00:15:00
#SBATCH --job-name=mcb-smoke
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM/logs/slurm/training/%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM/logs/slurm/training/%x-%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --environment=/capstor/store/cscs/swissai/a06/containers/NGC-PyTorch/ngc_pt_jan.toml
#SBATCH --no-requeue

# Multi-codebook smoke test on 2 nodes (8 GPUs):
#   TP=2 x PP=1 x CP=2 x DP=2 = 8 GPUs
#   Sequence parallel ON
#   Tiny model + mock data (NullTokenizer)
#   20 training steps with one mid-training checkpoint save
#
# Goal: exercise every parallelism dimension at once and confirm the
#       multi-codebook heads + input embeddings + loss + audio_pad path
#       all run without crashing and produce sane per-head losses.
#
# Pass criteria:
#   - Job completes 20 iters without NaN/Inf
#   - All 6 metrics logged: lm loss, text loss, audio loss mean,
#     audio loss k0..k3
#   - audio loss k0 token count > audio loss k3 count (pad triangle visible)
#   - Checkpoint save+load roundtrip works

echo "START TIME: $(date)"

################ Configs ################

# Parallelism (TP * PP * CP * DP must equal total GPUs = NNODES * 4)
TP_SIZE=2
PP_SIZE=1
CP_SIZE=2

# Tiny model dims for fast iteration. Constraints:
#   hidden_size, ffn_hidden_size, num_attention_heads divisible by TP
#   num_query_groups divisible by TP
#   num_layers divisible by PP
#   seq_length divisible by 2 * CP
NUM_LAYERS=4
HIDDEN_SIZE=256
FFN_HIDDEN_SIZE=512
NUM_ATTN_HEADS=4
NUM_QUERY_GROUPS=2
SEQ_LEN=128
TEXT_VOCAB_SIZE=1024

# Multi-codebook config (HCodec-1.0-like, scaled down)
NUM_AUDIO_CB=4
AUDIO_VOCAB=128          # small for fast TP-divisibility and quick lookups
AUDIO_PAD_ID=127         # last index reserved for <audio_pad>
AUDIO_LOSS_WEIGHT=1.0

MBS=2
GBS=4                    # MBS * DP * grad_accum = 2 * 2 * 1 = 4  (DP=8/(TP*PP*CP)=2)
TRAINING_STEPS=20
CHECKPOINT_STEPS=10      # save once mid-run to exercise checkpoint path

#### Debugging ####
LOG_NCCL=false
###################

# Paths
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM

PROJECT_NAME=Megatron-Clariden-MultiCodebook-Smoke
EXP_NAME=mcb-smoke-${SLURM_NNODES}n-tp${TP_SIZE}-pp${PP_SIZE}-cp${CP_SIZE}-sp
PROJECT_DIR=$MEGATRON_LM_DIR/logs/Meg-Runs/$PROJECT_NAME

#########################################

EXP_DIR=$PROJECT_DIR/$EXP_NAME
CKPT_DIR=$EXP_DIR/checkpoints
TRIGGER_DIR=$EXP_DIR/triggers
DEBUG_DIR=$EXP_DIR/debug/$SLURM_JOB_ID
COMPUTE_ENVIRONMENT_DIR=$DEBUG_DIR/compute_environment.txt
GPU_MEM_LOGGING=$DEBUG_DIR/memory_logging.txt
LOGGING_DIR=$EXP_DIR/logging
TENSORBOARD_DIR=$LOGGING_DIR/tensorboard

# Distributed env
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=6000
export WORLD_SIZE=$SLURM_NPROCS

ulimit -c 0

#### Megatron Args ####

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
	--make-vocab-size-divisible-by 16
	--normalization RMSNorm
	--swiglu
	--untie-embeddings-and-output-weights
)

# Multi-codebook RVQ heads (the feature under test)
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
	--tensorboard-dir $TENSORBOARD_DIR
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
	--lr-warmup-iters 5
)

CHECKPOINTING_ARGS=(
	--save $CKPT_DIR
	--save-interval $CHECKPOINT_STEPS
	--ckpt-format torch_dist
	--load $CKPT_DIR
	--trigger-path $TRIGGER_DIR
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
)

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

# Set up directories
mkdir -p $CKPT_DIR
mkdir -p $TRIGGER_DIR
mkdir -p $PROJECT_DIR
mkdir -p $DEBUG_DIR
mkdir -p $LOGGING_DIR

echo "[$(date)] Using codebase in $MEGATRON_LM_DIR"
echo "[$(date)] TP=$TP_SIZE PP=$PP_SIZE CP=$CP_SIZE SP=on  (NPROCS=$WORLD_SIZE)"

cd $MEGATRON_LM_DIR
export PYTHONPATH=$MEGATRON_LM_DIR:$PYTHONPATH

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

# NCCL Debug
if [ "$LOG_NCCL" = true ]; then
  CMD_PREFIX="NCCL_DEBUG=INFO NCCL_DEBUG_FILE=$DEBUG_DIR/nccl-info-hostname-\$SLURMD_NODENAME-local-rank-\$SLURM_LOCALID-procid-\$SLURM_PROCID.txt $CMD_PREFIX"
fi

# Save sbatch script
cp $0 $DEBUG_DIR

# Compute environment snapshot
echo -e "$(date)" > $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR
echo -e "\nCMD: $CMD_PREFIX $TRAINING_CMD" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR
echo -e "\nNODES: $(scontrol show hostnames $SLURM_JOB_NODELIST)" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR
echo -e "\nMegatron path: $MEGATRON_LM_DIR ($(git -C $MEGATRON_LM_DIR rev-parse --verify HEAD))" >> $COMPUTE_ENVIRONMENT_DIR

srun -lu bash -c 'echo $(hostname) $(nvidia-smi | grep -o "|\\s*[0-9]*MiB")' > $GPU_MEM_LOGGING

srun --cpus-per-task $SLURM_CPUS_PER_TASK -lu bash -c "
  export LD_LIBRARY_PATH=\$(echo \$LD_LIBRARY_PATH | tr ':' '\n' | grep -v compat | paste -sd ':' -)
  if [ -d /usr/local/cuda/compat ]; then mv /usr/local/cuda/compat /usr/local/cuda/compat_disabled 2>/dev/null || true; fi
  RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID $CMD_PREFIX $TRAINING_CMD"

echo "END TIME: $(date)"
