#!/bin/bash
#
# Phase 2: 评估对比
#
# 目标: 对比 baseline vs memory-enhanced 的性能
# 包含三种评估:
#   1. Baseline (无 memory)
#   2. Always trigger (memory enabled, trigger always on)
#   3. Learned trigger (memory enabled, trigger learned)
#

set -e

export DEBUG_MODE=false
export CUDA_VISIBLE_DEVICES=0
export MAIN_PROCESS_PORT=29507
export HF_HOME=/root/autodl-tmp/models
export TRANSFORMERS_CACHE=/root/autodl-tmp/models
export HF_DATASETS_CACHE=/root/autodl-tmp/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0

# 模型配置
REASONER_MODEL="/root/autodl-tmp/models/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
WEAVER_MODEL="/root/autodl-tmp/models/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
TRIGGER_MODEL="/root/autodl-tmp/models/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"

# 数据集配置
DATASET_NAME="triviaqa"

# Phase 2 配置
EXPERIENCE_STORE_PATH="/root/autodl-tmp/phase0_experience.jsonl"
WEAVER_CHECKPOINT="results/train/triviaqa/root/pn=2_pl=2_in=0_il=2_20260121-220038/weaver"
TRIGGER_CHECKPOINT="trainer_output/phase1b_trigger"

EVAL_BATCH_SIZE=1

# Augmentation configs (for TriviaQA)
MAX_PROMPT_AUG_NUM=2
MAX_INFERENCE_AUG_NUM=0
PROMPT_LATENTS_LEN=2
INFERENCE_LATENTS_LEN=2

# Memory 配置
MEMORY_TOPK=1
MEMORY_MIN_SCORE=0.3

echo "================================================================================"
echo "Phase 2: 评估对比"
echo "================================================================================"
echo ""
echo "配置信息:"
echo "  - 模型: $REASONER_MODEL"
echo "  - 数据集: $DATASET_NAME"
echo "  - 经验库: $EXPERIENCE_STORE_PATH"
echo ""
echo "评估方案:"
echo "  1. Baseline (无 memory)"
echo "  2. Always trigger (memory enabled, trigger always on)"
echo "  3. Learned trigger (memory enabled, trigger learned)"
echo "================================================================================"
echo ""

# ============================================================================
# 1. Baseline 评估 (无 memory)
# ============================================================================
echo "================================================================================"
echo "1. 评估 Baseline (无 memory)"
echo "================================================================================"
echo ""

read -p "是否评估 Baseline? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "开始评估 Baseline..."
    echo ""

    echo "=== 评估前 GPU 状态 ==="
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits
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
        dataset.name ${DATASET_NAME} \
        dataset.mode sft \
        run.mode evaluate \
        run.generation.eval_batch_size ${EVAL_BATCH_SIZE} \
        memory.enable false

    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Baseline 评估完成"
        echo ""
        echo "=== 评估后 GPU 状态 ==="
        nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits
        echo ""
        # 清理 GPU 显存
        sleep 2
        python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
        echo "GPU 显存已清理"
        echo "=== 清理后 GPU 状态 ==="
        nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits
        echo ""
    else
        echo "✗ Baseline 评估失败"
        exit 1
    fi
else
    echo "✓ 跳过 Baseline 评估"
    echo ""
fi

# 清理 GPU 显存
sleep 2
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# ============================================================================
# 2. Always Trigger 评估 (memory enabled, trigger always on)
# ============================================================================
echo "================================================================================"
echo "2. 评估 Always Trigger (memory enabled, trigger always on)"
echo "================================================================================"
echo ""

# 检查经验库
if [ ! -f "$EXPERIENCE_STORE_PATH" ]; then
    echo "⚠️  警告: 经验库不存在: $EXPERIENCE_STORE_PATH"
    echo "跳过 memory-enhanced 评估"
    echo ""
else
    NUM_EXPERIENCES=$(wc -l < "$EXPERIENCE_STORE_PATH")
    echo "经验库统计:"
    echo "  - 条目数: $NUM_EXPERIENCES"
    echo ""

    read -p "是否评估 Always Trigger? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # 确定使用哪个 checkpoint
        if [ -d "$WEAVER_CHECKPOINT" ]; then
            MODEL_PATH=$WEAVER_CHECKPOINT
            echo "使用 Weaver checkpoint: $WEAVER_CHECKPOINT"
        else
            MODEL_PATH=null
            echo "⚠️  Weaver checkpoint 不存在，使用 baseline 模型"
        fi

        echo "开始评估 Always Trigger..."
        echo ""

        echo "=== 评估前 GPU 状态 ==="
        nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits
        echo ""

        python main.py \
            --cfg-path configs/latent_memory/phase2_memory_eval.yaml \
            --options \
            model.model_name ${REASONER_MODEL} \
            model.load_model_path ${MODEL_PATH} \
            model.max_prompt_aug_num ${MAX_PROMPT_AUG_NUM} \
            model.max_inference_aug_num ${MAX_INFERENCE_AUG_NUM} \
            model.weaver.model_name ${WEAVER_MODEL} \
            model.weaver.prompt_latents_len ${PROMPT_LATENTS_LEN} \
            model.weaver.inference_latents_len ${INFERENCE_LATENTS_LEN} \
            model.trigger.model_name ${TRIGGER_MODEL} \
            model.trigger.active False \
            dataset.name ${DATASET_NAME} \
            dataset.mode sft \
            run.mode evaluate \
            run.generation.eval_batch_size ${EVAL_BATCH_SIZE} \
            memory.enable true \
            memory.store_path ${EXPERIENCE_STORE_PATH} \
            memory.index_type simple \
            memory.topk ${MEMORY_TOPK} \
            memory.min_score ${MEMORY_MIN_SCORE} \
            memory.writeback.enable false

        if [ $? -eq 0 ]; then
            echo ""
            echo "✓ Always Trigger 评估完成"
            echo ""
            echo "=== 评估后 GPU 状态 ==="
            nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits
            echo ""
            # 清理 GPU 显存
            sleep 2
            python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
            echo "GPU 显存已清理"
            echo "=== 清理后 GPU 状态 ==="
            nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits
            echo ""
        else
            echo "✗ Always Trigger 评估失败"
            exit 1
        fi
    else
        echo "✓ 跳过 Always Trigger 评估"
        echo ""
    fi
fi

# 清理 GPU 显存
sleep 2
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# ============================================================================
# 3. Learned Trigger 评估 (memory enabled, trigger learned)
# ============================================================================
echo "================================================================================"
echo "3. 评估 Learned Trigger (memory enabled, trigger learned)"
echo "================================================================================"
echo ""

if [ ! -f "$EXPERIENCE_STORE_PATH" ]; then
    echo "⚠️  警告: 经验库不存在，跳过评估"
    echo ""
elif [ ! -d "$TRIGGER_CHECKPOINT" ]; then
    echo "⚠️  警告: Trigger checkpoint 不存在: $TRIGGER_CHECKPOINT"
    echo "请先运行 Phase 1b 训练 Trigger:"
    echo "  bash scripts/phase1b_train_trigger.sh"
    echo ""
else
    read -p "是否评估 Learned Trigger? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "使用 Trigger checkpoint: $TRIGGER_CHECKPOINT"
        echo "开始评估 Learned Trigger..."
        echo ""

        python main.py \
            --cfg-path configs/latent_memory/phase2_memory_eval.yaml \
            --options \
            model.model_name ${REASONER_MODEL} \
            model.load_model_path ${TRIGGER_CHECKPOINT} \
            model.max_prompt_aug_num ${MAX_PROMPT_AUG_NUM} \
            model.max_inference_aug_num ${MAX_INFERENCE_AUG_NUM} \
            model.weaver.model_name ${WEAVER_MODEL} \
            model.weaver.prompt_latents_len ${PROMPT_LATENTS_LEN} \
            model.weaver.inference_latents_len ${INFERENCE_LATENTS_LEN} \
            model.trigger.model_name ${TRIGGER_MODEL} \
            model.trigger.active True \
            dataset.name ${DATASET_NAME} \
            dataset.mode sft \
            run.mode evaluate \
            run.generation.eval_batch_size ${EVAL_BATCH_SIZE} \
            memory.enable true \
            memory.store_path ${EXPERIENCE_STORE_PATH} \
            memory.index_type simple \
            memory.topk ${MEMORY_TOPK} \
            memory.min_score ${MEMORY_MIN_SCORE} \
            memory.writeback.enable false

        if [ $? -eq 0 ]; then
            echo ""
            echo "✓ Learned Trigger 评估完成"
            echo ""
        else
            echo "✗ Learned Trigger 评估失败"
            exit 1
        fi
    else
        echo "✓ 跳过 Learned Trigger 评估"
        echo ""
    fi
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "================================================================================"
echo "Phase 2 评估完成"
echo "================================================================================"
echo ""
echo "评估结果对比:"
echo "  1. Baseline (无 memory)"
echo "  2. Always trigger (memory enabled, trigger always on)"
echo "  3. Learned trigger (memory enabled, trigger learned)"
echo ""
echo "查看详细结果:"
echo "  - 查看日志文件"
echo "  - 对比 EM 和 F1 指标"
echo ""
echo "消融实验建议:"
echo "  - Baseline vs Always trigger: 验证 memory 是否有用"
echo "  - Always trigger vs Learned trigger: 验证 trigger 是否学到了有效策略"
echo ""
echo "================================================================================"
