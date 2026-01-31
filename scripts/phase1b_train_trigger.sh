#!/bin/bash
#
# Phase 1b: 训练 Trigger (GRPO)
#
# 目标: 训练 Trigger 学习何时触发 Weaver 生成 latent memory
# 配置: GRPO training + memory enabled + trigger active + weaver frozen
#

set -e

export DEBUG_MODE=false
export CUDA_VISIBLE_DEVICES=0
export MAIN_PROCESS_PORT=29507
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export HF_HOME=/root/autodl-tmp/models
export TRANSFORMERS_CACHE=/root/autodl-tmp/models
export HF_DATASETS_CACHE=/root/autodl-tmp/datasets
export WANDB_PROJECT=self-evolve-rag
export WANDB_LOG_MODEL=false

# 模型配置
REASONER_MODEL="/root/autodl-tmp/models/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
WEAVER_MODEL="/root/autodl-tmp/models/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
TRIGGER_MODEL="/root/autodl-tmp/models/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"

# 数据集配置
DATASET_NAME="triviaqa"

# Phase 1b 配置
EXPERIENCE_STORE_PATH="/root/autodl-tmp/phase0_experience.jsonl"
WEAVER_CHECKPOINT="results/train/triviaqa/root/pn=2_pl=2_in=0_il=2_20260121-220038/weaver"
OUTPUT_DIR="trainer_output/phase1b_trigger"

# 训练超参数
NUM_EPOCHS=1
BATCH_SIZE=1
GRAD_ACCUM_STEPS=4
LEARNING_RATE=5e-6
NUM_GENERATIONS=4  # GRPO 每个 prompt 生成多少个 response

# Memory 配置
MEMORY_TOPK=1
MEMORY_MIN_SCORE=0.3
WRITEBACK_MIN_REWARD=0.6

# Augmentation configs (for TriviaQA)
MAX_PROMPT_AUG_NUM=2
MAX_INFERENCE_AUG_NUM=0
PROMPT_LATENTS_LEN=2
INFERENCE_LATENTS_LEN=2

echo "================================================================================"
echo "Phase 1b: 训练 Trigger (GRPO)"
echo "================================================================================"
echo ""
echo "配置信息:"
echo "  - 模型: $REASONER_MODEL"
echo "  - 数据集: $DATASET_NAME"
echo "  - 经验库: $EXPERIENCE_STORE_PATH"
echo "  - Weaver checkpoint: $WEAVER_CHECKPOINT"
echo "  - 输出目录: $OUTPUT_DIR"
echo ""
echo "训练配置:"
echo "  - 训练方法: GRPO (强化学习)"
echo "  - 训练轮数: $NUM_EPOCHS"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Gradient accumulation: $GRAD_ACCUM_STEPS"
echo "  - Learning rate: $LEARNING_RATE"
echo "  - Num generations: $NUM_GENERATIONS"
echo ""
echo "Memory 配置:"
echo "  - Trigger: active (学习何时触发)"
echo "  - Weaver: frozen (不训练)"
echo "  - Memory retrieval: enabled"
echo "  - Writeback: enabled"
echo "================================================================================"
echo ""

# 检查 Weaver checkpoint 是否存在
if [ ! -d "$WEAVER_CHECKPOINT" ]; then
    echo "✗ 错误: Weaver checkpoint 不存在: $WEAVER_CHECKPOINT"
    echo ""
    echo "请先运行 Phase 1a 训练 Weaver:"
    echo "  bash scripts/phase1a_train_weaver.sh"
    echo ""
    exit 1
fi

# 检查经验库是否存在
if [ ! -f "$EXPERIENCE_STORE_PATH" ]; then
    echo "✗ 错误: 经验库不存在: $EXPERIENCE_STORE_PATH"
    echo ""
    echo "请先运行 Phase 0 生成经验库:"
    echo "  bash scripts/phase0_coldstart.sh"
    echo ""
    exit 1
fi

NUM_EXPERIENCES=$(wc -l < "$EXPERIENCE_STORE_PATH")
echo "经验库统计:"
echo "  - 条目数: $NUM_EXPERIENCES"
echo ""

read -p "是否开始训练? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "✓ 取消训练"
    exit 0
fi

echo "开始训练 Trigger..."
echo ""

# 运行 Phase 1b 训练
python -m accelerate.commands.launch \
    --config_file=configs/single_gpu.yaml \
    main.py \
    --cfg-path configs/latent_memory/${DATASET_NAME}.yaml \
    --options \
    model.model_name ${REASONER_MODEL} \
    model.load_model_path ${WEAVER_CHECKPOINT} \
    model.max_prompt_aug_num ${MAX_PROMPT_AUG_NUM} \
    model.max_inference_aug_num ${MAX_INFERENCE_AUG_NUM} \
    model.weaver.model_name ${WEAVER_MODEL} \
    model.weaver.prompt_latents_len ${PROMPT_LATENTS_LEN} \
    model.weaver.inference_latents_len ${INFERENCE_LATENTS_LEN} \
    model.trigger.model_name ${TRIGGER_MODEL} \
    model.trigger.active True \
    dataset.name ${DATASET_NAME} \
    dataset.mode grpo \
    run.mode train \
    run.train_weaver False \
    run.train_trigger True \
    run.train_trigger_method grpo \
    run.trigger.grpo.num_train_epochs ${NUM_EPOCHS} \
    run.trigger.grpo.per_device_train_batch_size ${BATCH_SIZE} \
    run.trigger.grpo.per_device_eval_batch_size ${BATCH_SIZE} \
    run.trigger.grpo.gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
    run.trigger.grpo.learning_rate ${LEARNING_RATE} \
    run.trigger.grpo.num_generations ${NUM_GENERATIONS} \
    run.trigger.grpo.output_dir ${OUTPUT_DIR} \
    run.trigger.grpo.logging_steps 10 \
    run.trigger.grpo.eval_steps 50 \
    run.trigger.grpo.save_steps 50 \
    memory.enable true \
    memory.store_path ${EXPERIENCE_STORE_PATH} \
    memory.index_type simple \
    memory.topk ${MEMORY_TOPK} \
    memory.min_score ${MEMORY_MIN_SCORE} \
    memory.writeback.enable true \
    memory.writeback.min_reward ${WRITEBACK_MIN_REWARD} \
    memory.writeback.require_grounding false

# 检查结果
if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "✓ Phase 1b 完成"
    echo "================================================================================"
    echo ""
    echo "输出文件:"
    echo "  - Trigger checkpoint: $OUTPUT_DIR"
    echo "  - 经验库 (更新后): $EXPERIENCE_STORE_PATH"
    echo ""

    if [ -f "$EXPERIENCE_STORE_PATH" ]; then
        NUM_EXPERIENCES_AFTER=$(wc -l < "$EXPERIENCE_STORE_PATH")
        echo "经验库统计:"
        echo "  - 训练前: $NUM_EXPERIENCES"
        echo "  - 训练后: $NUM_EXPERIENCES_AFTER"
        echo "  - 新增: $((NUM_EXPERIENCES_AFTER - NUM_EXPERIENCES))"
        echo ""
    fi

    echo "查看训练日志:"
    echo "  tensorboard --logdir $OUTPUT_DIR"
    echo ""
    echo "下一步:"
    echo "  运行 Phase 2 评估对比:"
    echo "  bash scripts/phase2_eval.sh"
    echo ""
    echo "================================================================================"
else
    echo ""
    echo "✗ Phase 1b 训练失败"
    exit 1
fi
