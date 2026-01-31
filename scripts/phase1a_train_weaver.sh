#!/bin/bash
#
# Phase 1a: 训练 Weaver (SFT)
#
# 目标: 训练 Weaver 学习如何融合外部经验并生成 latent memory
# 配置: SFT training + memory enabled + trigger always on (active=False)
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

# Phase 1a 配置
EXPERIENCE_STORE_PATH="/root/autodl-tmp/phase0_experience.jsonl"
OUTPUT_DIR="trainer_output/phase1a_weaver"

# 训练超参数
NUM_EPOCHS=2
BATCH_SIZE=1
GRAD_ACCUM_STEPS=4
LEARNING_RATE=5e-6
MAX_LENGTH=1024

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
echo "Phase 1a: 训练 Weaver (SFT)"
echo "================================================================================"
echo ""
echo "配置信息:"
echo "  - 模型: $REASONER_MODEL"
echo "  - 数据集: $DATASET_NAME"
echo "  - 经验库: $EXPERIENCE_STORE_PATH"
echo "  - 输出目录: $OUTPUT_DIR"
echo ""
echo "训练配置:"
echo "  - 训练方法: SFT"
echo "  - 训练轮数: $NUM_EPOCHS"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Gradient accumulation: $GRAD_ACCUM_STEPS"
echo "  - Learning rate: $LEARNING_RATE"
echo ""
echo "Memory 配置:"
echo "  - Trigger: always on (active=False)"
echo "  - Memory retrieval: enabled"
echo "  - Writeback: enabled (继续积累经验)"
echo "  - Top-k: $MEMORY_TOPK"
echo "  - Min score: $MEMORY_MIN_SCORE"
echo "================================================================================"
echo ""

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

echo "开始训练 Weaver..."
echo ""

# 运行 Phase 1a 训练
python -m accelerate.commands.launch \
    --config_file=configs/single_gpu.yaml \
    main.py \
    --cfg-path configs/latent_memory/phase1_real_training.yaml \
    --options \
    model.model_name ${REASONER_MODEL} \
    model.load_model_path null \
    model.max_prompt_aug_num ${MAX_PROMPT_AUG_NUM} \
    model.max_inference_aug_num ${MAX_INFERENCE_AUG_NUM} \
    model.weaver.model_name ${WEAVER_MODEL} \
    model.weaver.prompt_latents_len ${PROMPT_LATENTS_LEN} \
    model.weaver.inference_latents_len ${INFERENCE_LATENTS_LEN} \
    model.trigger.model_name ${TRIGGER_MODEL} \
    model.trigger.active False \
    dataset.name ${DATASET_NAME} \
    dataset.mode sft \
    run.mode train \
    run.train_weaver True \
    run.train_weaver_method sft \
    run.train_trigger False \
    run.weaver.sft.num_train_epochs ${NUM_EPOCHS} \
    run.weaver.sft.per_device_train_batch_size ${BATCH_SIZE} \
    run.weaver.sft.per_device_eval_batch_size ${BATCH_SIZE} \
    run.weaver.sft.gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
    run.weaver.sft.learning_rate ${LEARNING_RATE} \
    run.weaver.sft.max_length ${MAX_LENGTH} \
    run.weaver.sft.bf16 True \
    run.weaver.sft.output_dir ${OUTPUT_DIR} \
    run.weaver.sft.logging_steps 10 \
    run.weaver.sft.eval_steps 100 \
    run.weaver.sft.save_steps 100 \
    run.weaver.sft.load_best_model_at_end True \
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
    echo "✓ Phase 1a 完成"
    echo "================================================================================"
    echo ""
    echo "输出文件:"
    echo "  - Weaver checkpoint: $OUTPUT_DIR"
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
    echo "  1. 直接评估 (Trigger always on):"
    echo "     bash scripts/phase2_eval.sh"
    echo ""
    echo "  2. 训练 Trigger (学习何时触发):"
    echo "     bash scripts/phase1b_train_trigger.sh"
    echo ""
    echo "================================================================================"
else
    echo ""
    echo "✗ Phase 1a 训练失败"
    exit 1
fi
