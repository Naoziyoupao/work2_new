#!/bin/bash
#SBATCH -J multiturn_cls
#SBATCH -o logs/train_%j.out
#SBATCH -e logs/train_%j.err
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=4:00:00

echo "=========================================="
echo "开始时间: $(date)"
echo "节点: $SLURM_NODELIST"
echo "任务ID: $SLURM_JOB_ID"
echo "=========================================="

# 交互式申请节点（调试用）：
# salloc -N 1 --mem 150GB -t 3:00:00 --gres=gpu:a100-sxm4-80gb:1

conda activate base

mkdir -p logs

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

WORK_DIR=/home/xzhang/work2_new
MODEL=/home/share/models/Llama-3.2-1B

# 修改 --train_file 切换训练轮次
python $WORK_DIR/train.py \
    --model_id $MODEL \
    --train_file $WORK_DIR/output/train_round1.jsonl \
    --val_file   $WORK_DIR/output/val.jsonl \
    --output_dir $WORK_DIR/outputs/exp001 \
    --epochs 3 \
    --batch_size 4 \
    --max_length 1024

echo "=========================================="
echo "结束时间: $(date)"
echo "=========================================="
