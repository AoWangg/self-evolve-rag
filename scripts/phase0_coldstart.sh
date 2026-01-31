#!/bin/bash
#
# Phase 0: 冷启动 - 生成初始经验库
#
# 目标: 使用 baseline 模型在数据集上生成回答，收集高质量经验
# 配置: eval mode + writeback enabled + memory disabled
#

set -e

export DEBUG_MODE=false
export CUDA_VISIBLE_DEVICES=0
export MAIN_PROCESS_PORT=29507
export HF_HOME=/root/autodl-tmp/models
export TRANSFORMERS_CACHE=/root/autodl-tmp/models
export HF_DATASETS_CACHE=/root/autodl-tmp/datasets

# 模型配置
REASONER_MODEL="/root/autodl-tmp/models/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
WEAVER_MODEL="/root/autodl-tmp/models/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
TRIGGER_MODEL="/root/autodl-tmp/models/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"

# 数据集配置
DATASET_NAME="triviaqa"

# Phase 0 配置
EXPERIENCE_STORE_PATH="/root/autodl-tmp/real_experience.jsonl"
MIN_REWARD=0.5  # 写回阈值（越低收集越多经验）
EVAL_BATCH_SIZE=1

# Augmentation configs (for TriviaQA)
MAX_PROMPT_AUG_NUM=2
MAX_INFERENCE_AUG_NUM=0
PROMPT_LATENTS_LEN=2
INFERENCE_LATENTS_LEN=2

echo "================================================================================"
echo "Phase 0: 冷启动 - 生成初始经验库"
echo "================================================================================"
echo ""
echo "配置信息:"
echo "  - 模型: $REASONER_MODEL"
echo "  - 数据集: $DATASET_NAME"
echo "  - 经验库路径: $EXPERIENCE_STORE_PATH"
echo "  - 最小 reward: $MIN_REWARD"
echo "  - Memory: disabled (不检索)"
echo "  - Writeback: enabled (收集经验)"
echo "================================================================================"
echo ""

# 检查经验库是否已存在
if [ -f "$EXPERIENCE_STORE_PATH" ]; then
    echo "⚠️  经验库已存在: $EXPERIENCE_STORE_PATH"
    NUM_EXISTING=$(wc -l < "$EXPERIENCE_STORE_PATH")
    echo "  - 现有条目数: $NUM_EXISTING"
    echo ""
    read -p "是否重新生成? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -f "$EXPERIENCE_STORE_PATH"
        echo "✓ 已删除旧经验库"
    else
        echo "✓ 退出，保留现有经验库"
        exit 0
    fi
fi

echo "开始生成初始经验库..."
echo ""

# 运行 Phase 0
python main.py \
    --cfg-path configs/latent_memory/phase0_real.yaml \
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
    run.train_weaver False \
    run.train_trigger False \
    run.generation.eval_batch_size ${EVAL_BATCH_SIZE} \
    memory.enable false \
    memory.store_path ${EXPERIENCE_STORE_PATH} \
    memory.writeback.enable true \
    memory.writeback.min_reward ${MIN_REWARD} \
    memory.writeback.require_grounding false

# 检查结果
if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "✓ Phase 0 完成"
    echo "================================================================================"

    if [ -f "$EXPERIENCE_STORE_PATH" ]; then
        NUM_EXPERIENCES=$(wc -l < "$EXPERIENCE_STORE_PATH")
        echo ""
        echo "经验库统计:"
        echo "  - 路径: $EXPERIENCE_STORE_PATH"
        echo "  - 总条目数: $NUM_EXPERIENCES"
        echo ""
        echo "查看经验库内容:"
        echo "  head -5 $EXPERIENCE_STORE_PATH"
        echo ""
        echo "下一步:"
        echo "  运行 Phase 1a 训练 Weaver:"
        echo "  bash scripts/phase1a_train_weaver.sh"
    else
        echo ""
        echo "⚠️  警告: 经验库文件未生成"
    fi
    echo ""
    echo "================================================================================"
else
    echo ""
    echo "✗ Phase 0 失败"
    exit 1
fi
