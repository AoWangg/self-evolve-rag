#!/bin/bash
#
# Self-Evolving RAG 完整训练流程
#
# 三阶段训练:
# Phase 0: 冷启动 - 生成初始经验库（eval mode + writeback）
# Phase 1a: 训练 Weaver - 使用 SFT 训练 Weaver（trigger always on）
# Phase 1b: 训练 Trigger - 使用 GRPO 训练 Trigger（weaver frozen）
# Phase 2: 评估对比 - baseline vs memory-enhanced
#

set -e  # Exit on error

# ============================================================================
# 环境配置
# ============================================================================
export DEBUG_MODE=false
export CUDA_VISIBLE_DEVICES=0
export MAIN_PROCESS_PORT=29507

# 模型配置
REASONER_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
WEAVER_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
TRIGGER_MODEL="Qwen/Qwen2.5-1.5B-Instruct"

# 数据集配置
DATASET_NAME="triviaqa"

# 路径配置
EXPERIENCE_STORE_PATH="/root/autodl-tmp/real_experience.jsonl"
WEAVER_CHECKPOINT_DIR="trainer_output/weaver_checkpoint"
TRIGGER_CHECKPOINT_DIR="trainer_output/trigger_checkpoint"

# 训练超参数
PHASE0_NUM_SAMPLES=500
PHASE1A_EPOCHS=2
PHASE1B_EPOCHS=1
BATCH_SIZE=1
GRAD_ACCUM_STEPS=4

# Augmentation configs (for TriviaQA)
MAX_PROMPT_AUG_NUM=2
MAX_INFERENCE_AUG_NUM=0
PROMPT_LATENTS_LEN=2
INFERENCE_LATENTS_LEN=2

echo "================================================================================"
echo "Self-Evolving RAG 完整训练流程"
echo "================================================================================"
echo ""
echo "配置信息:"
echo "  - 模型: $REASONER_MODEL"
echo "  - 数据集: $DATASET_NAME"
echo "  - 经验库路径: $EXPERIENCE_STORE_PATH"
echo "  - Phase 0 样本数: $PHASE0_NUM_SAMPLES"
echo "  - Phase 1a 训练轮数: $PHASE1A_EPOCHS"
echo "  - Phase 1b 训练轮数: $PHASE1B_EPOCHS"
echo "================================================================================"
echo ""

# ============================================================================
# Phase 0: 冷启动 - 生成初始经验库
# ============================================================================
echo "================================================================================"
echo "Phase 0: 冷启动 - 生成初始经验库"
echo "================================================================================"
echo ""

if [ -f "$EXPERIENCE_STORE_PATH" ]; then
    echo "⚠️  经验库已存在: $EXPERIENCE_STORE_PATH"
    read -p "是否重新生成? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -f "$EXPERIENCE_STORE_PATH"
        echo "✓ 已删除旧经验库"
    else
        echo "✓ 跳过 Phase 0，使用现有经验库"
        SKIP_PHASE0=true
    fi
fi

if [ "$SKIP_PHASE0" != "true" ]; then
    echo "开始生成初始经验库..."
    echo ""

    python main.py \
        --cfg-path configs/latent_memory/phase0_real.yaml \
        --options \
        model.model_name ${REASONER_MODEL} \
        model.max_prompt_aug_num ${MAX_PROMPT_AUG_NUM} \
        model.max_inference_aug_num ${MAX_INFERENCE_AUG_NUM} \
        model.weaver.model_name ${WEAVER_MODEL} \
        model.weaver.prompt_latents_len ${PROMPT_LATENTS_LEN} \
        model.weaver.inference_latents_len ${INFERENCE_LATENTS_LEN} \
        model.trigger.model_name ${TRIGGER_MODEL} \
        model.trigger.active False \
        run.mode eval \
        run.train_weaver False \
        run.train_trigger False \
        memory.enable false \
        memory.store_path ${EXPERIENCE_STORE_PATH} \
        memory.writeback.enable true \
        memory.writeback.min_reward 0.5 \
        memory.writeback.require_grounding false

    if [ $? -eq 0 ]; then
        NUM_EXPERIENCES=$(wc -l < "$EXPERIENCE_STORE_PATH" 2>/dev/null || echo "0")
        echo ""
        echo "✓ Phase 0 完成"
        echo "  - 生成经验数: $NUM_EXPERIENCES"
        echo ""
    else
        echo "✗ Phase 0 失败"
        exit 1
    fi
fi

# ============================================================================
# Phase 1a: 训练 Weaver (SFT)
# ============================================================================
echo "================================================================================"
echo "Phase 1a: 训练 Weaver (SFT)"
echo "================================================================================"
echo ""
echo "配置:"
echo "  - Trigger: always on (active=False)"
echo "  - Memory: enabled (检索经验库)"
echo "  - 训练方法: SFT"
echo "  - 训练轮数: $PHASE1A_EPOCHS"
echo "================================================================================"
echo ""

read -p "是否开始 Phase 1a 训练? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "开始训练 Weaver..."
    echo ""

    python -m accelerate.commands.launch \
        --config_file=configs/zero2.yaml \
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
        dataset.mode sft \
        run.mode train \
        run.train_weaver True \
        run.train_weaver_method sft \
        run.train_trigger False \
        run.weaver.sft.num_train_epochs ${PHASE1A_EPOCHS} \
        run.weaver.sft.per_device_train_batch_size ${BATCH_SIZE} \
        run.weaver.sft.gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
        run.weaver.sft.learning_rate 5e-6 \
        run.weaver.sft.bf16 True \
        run.weaver.sft.output_dir ${WEAVER_CHECKPOINT_DIR} \
        memory.enable true \
        memory.store_path ${EXPERIENCE_STORE_PATH} \
        memory.index_type simple \
        memory.topk 1 \
        memory.min_score 0.3 \
        memory.writeback.enable true \
        memory.writeback.min_reward 0.6 \
        memory.writeback.require_grounding false

    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Phase 1a 完成"
        echo "  - Weaver checkpoint: $WEAVER_CHECKPOINT_DIR"
        echo ""
    else
        echo "✗ Phase 1a 失败"
        exit 1
    fi
else
    echo "✓ 跳过 Phase 1a"
    echo ""
fi

# ============================================================================
# Phase 1b: 训练 Trigger (GRPO)
# ============================================================================
echo "================================================================================"
echo "Phase 1b: 训练 Trigger (GRPO)"
echo "================================================================================"
echo ""
echo "配置:"
echo "  - Weaver: frozen (加载 Phase 1a checkpoint)"
echo "  - Trigger: active=True (学习何时触发)"
echo "  - Memory: enabled"
echo "  - 训练方法: GRPO"
echo "  - 训练轮数: $PHASE1B_EPOCHS"
echo "================================================================================"
echo ""

read -p "是否开始 Phase 1b 训练? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "开始训练 Trigger..."
    echo ""

    # 检查 Weaver checkpoint 是否存在
    if [ ! -d "$WEAVER_CHECKPOINT_DIR" ]; then
        echo "⚠️  警告: Weaver checkpoint 不存在: $WEAVER_CHECKPOINT_DIR"
        echo "请先运行 Phase 1a 或指定正确的 checkpoint 路径"
        exit 1
    fi

    python -m accelerate.commands.launch \
        --config_file=configs/zero2.yaml \
        main.py \
        --cfg-path configs/latent_memory/${DATASET_NAME}.yaml \
        --options \
        model.model_name ${REASONER_MODEL} \
        model.load_model_path ${WEAVER_CHECKPOINT_DIR} \
        model.max_prompt_aug_num ${MAX_PROMPT_AUG_NUM} \
        model.max_inference_aug_num ${MAX_INFERENCE_AUG_NUM} \
        model.weaver.model_name ${WEAVER_MODEL} \
        model.weaver.prompt_latents_len ${PROMPT_LATENTS_LEN} \
        model.weaver.inference_latents_len ${INFERENCE_LATENTS_LEN} \
        model.trigger.model_name ${TRIGGER_MODEL} \
        model.trigger.active True \
        dataset.mode grpo \
        run.mode train \
        run.train_weaver False \
        run.train_trigger True \
        run.train_trigger_method grpo \
        run.trigger.grpo.num_train_epochs ${PHASE1B_EPOCHS} \
        run.trigger.grpo.per_device_train_batch_size ${BATCH_SIZE} \
        run.trigger.grpo.gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
        run.trigger.grpo.learning_rate 5e-6 \
        run.trigger.grpo.num_generations 4 \
        run.trigger.grpo.output_dir ${TRIGGER_CHECKPOINT_DIR} \
        memory.enable true \
        memory.store_path ${EXPERIENCE_STORE_PATH} \
        memory.index_type simple \
        memory.topk 1 \
        memory.min_score 0.3 \
        memory.writeback.enable true \
        memory.writeback.min_reward 0.6 \
        memory.writeback.require_grounding false

    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Phase 1b 完成"
        echo "  - Trigger checkpoint: $TRIGGER_CHECKPOINT_DIR"
        echo ""
    else
        echo "✗ Phase 1b 失败"
        exit 1
    fi
else
    echo "✓ 跳过 Phase 1b"
    echo ""
fi

# ============================================================================
# Phase 2: 评估对比
# ============================================================================
echo "================================================================================"
echo "Phase 2: 评估对比"
echo "================================================================================"
echo ""

read -p "是否开始 Phase 2 评估? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Baseline evaluation (no memory)
    echo ""
    echo "----------------------------------------"
    echo "评估 Baseline (无 memory)"
    echo "----------------------------------------"
    echo ""

    python main.py \
        --cfg-path configs/latent_memory/phase2_baseline_eval.yaml \
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
        run.mode eval \
        memory.enable false

    if [ $? -ne 0 ]; then
        echo "✗ Baseline 评估失败"
        exit 1
    fi

    # Memory-enhanced evaluation (with learned trigger)
    echo ""
    echo "----------------------------------------"
    echo "评估 Memory-Enhanced (with learned trigger)"
    echo "----------------------------------------"
    echo ""

    # 确定加载哪个 checkpoint
    if [ -d "$TRIGGER_CHECKPOINT_DIR" ]; then
        EVAL_MODEL_PATH=$TRIGGER_CHECKPOINT_DIR
        EVAL_TRIGGER_ACTIVE=True
        echo "使用 Trigger checkpoint: $TRIGGER_CHECKPOINT_DIR"
    elif [ -d "$WEAVER_CHECKPOINT_DIR" ]; then
        EVAL_MODEL_PATH=$WEAVER_CHECKPOINT_DIR
        EVAL_TRIGGER_ACTIVE=False
        echo "使用 Weaver checkpoint (Trigger always on): $WEAVER_CHECKPOINT_DIR"
    else
        echo "⚠️  警告: 没有找到训练好的模型，使用 baseline 模型"
        EVAL_MODEL_PATH=null
        EVAL_TRIGGER_ACTIVE=False
    fi

    python main.py \
        --cfg-path configs/latent_memory/phase2_memory_eval.yaml \
        --options \
        model.model_name ${REASONER_MODEL} \
        model.load_model_path ${EVAL_MODEL_PATH} \
        model.max_prompt_aug_num ${MAX_PROMPT_AUG_NUM} \
        model.max_inference_aug_num ${MAX_INFERENCE_AUG_NUM} \
        model.weaver.model_name ${WEAVER_MODEL} \
        model.weaver.prompt_latents_len ${PROMPT_LATENTS_LEN} \
        model.weaver.inference_latents_len ${INFERENCE_LATENTS_LEN} \
        model.trigger.model_name ${TRIGGER_MODEL} \
        model.trigger.active ${EVAL_TRIGGER_ACTIVE} \
        run.mode eval \
        memory.enable true \
        memory.store_path ${EXPERIENCE_STORE_PATH} \
        memory.index_type simple \
        memory.topk 1 \
        memory.min_score 0.3 \
        memory.writeback.enable false

    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Phase 2 评估完成"
        echo ""
    else
        echo "✗ Memory 评估失败"
        exit 1
    fi
else
    echo "✓ 跳过 Phase 2"
    echo ""
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "================================================================================"
echo "训练流程完成！"
echo "================================================================================"
echo ""
echo "生成的文件:"
echo "  - 经验库: $EXPERIENCE_STORE_PATH"
if [ -d "$WEAVER_CHECKPOINT_DIR" ]; then
    echo "  - Weaver checkpoint: $WEAVER_CHECKPOINT_DIR"
fi
if [ -d "$TRIGGER_CHECKPOINT_DIR" ]; then
    echo "  - Trigger checkpoint: $TRIGGER_CHECKPOINT_DIR"
fi
echo ""
echo "下一步:"
echo "  1. 查看经验库: head -10 $EXPERIENCE_STORE_PATH"
echo "  2. 查看训练日志: ls -lh trainer_output/"
echo "  3. 查看 TensorBoard: tensorboard --logdir trainer_output/"
echo ""
echo "消融实验建议:"
echo "  - Baseline (无 memory): trigger.active=False, memory.enable=false"
echo "  - Always trigger: trigger.active=False, memory.enable=true"
echo "  - Learned trigger: trigger.active=True, memory.enable=true"
echo ""
echo "================================================================================"
