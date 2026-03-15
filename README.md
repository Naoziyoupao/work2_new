# work2_new — 多轮对话来源分类器

用 oasst2（人类标注）和 Magpie（LLM 合成）两类多轮对话数据，训练一个二分类器，判断对话来源。

---

## 项目结构

```
.
├── extract_datasets.py   # 从 HuggingFace 下载并提取数据，生成 pool 和 round 1 训练集
├── sample_round2.py      # 第二轮采样（从 pool 排除已用 ID）
├── resample.py           # 通用重采样工具
├── train.py              # 训练脚本（LoRA + PoE 二分类器）
├── read.md               # 数据格式与字段详细说明
├── output/
│   ├── pool_oasst2_train.jsonl   # oasst2 train split 全量（17,427 条，无 label）
│   ├── pool_oasst2_val.jsonl     # oasst2 官方 validation split 全量（916 条，无 label）
│   ├── pool_magpie.jsonl         # Magpie 全量（300,000 条，无 label）
│   ├── train_round1.jsonl        # 第一轮训练集（10,000 条，含 label）
│   ├── train_round2.jsonl        # 第二轮训练集（11,000 条，含 label）
│   ├── val.jsonl                 # 验证集（1,000 条，含 label，固定不变）
│   └── selected_ids.json         # 已用 ID 记录（累计跨轮排除）
└── outputs/                      # 训练产物（checkpoints、报告、曲线图）
```

> `output/pool_*.jsonl` 体积较大（最大 1.7 GB），已加入 `.gitignore`，不入库。
> 首次使用需在本机运行 `extract_datasets.py` 生成。

---

## 环境依赖

```bash
pip install datasets transformers peft torch tqdm scikit-learn matplotlib
```

Python 推荐 3.10+，CUDA 推荐 11.8+。

---

## 快速开始

### 1. 生成数据集（首次运行）

从 HuggingFace 下载 oasst2 和 Magpie，提取英文多轮对话，生成第一轮训练集和验证集：

```bash
python extract_datasets.py
```

输出：
- `output/pool_oasst2_train.jsonl` — 17,427 条
- `output/pool_oasst2_val.jsonl` — 916 条
- `output/pool_magpie.jsonl` — 300,000 条
- `output/train_round1.jsonl` — 10,000 条（oasst2×5000 + Magpie×5000）
- `output/val.jsonl` — 1,000 条（oasst2×500 + Magpie×500，来自官方 val split）
- `output/selected_ids.json` — 已用 ID 记录

### 2. 训练模型

```bash
python train.py \
    --model_id /path/to/Llama-3.2-1B \
    --train_file output/train_round1.jsonl \
    --val_file output/val.jsonl \
    --output_dir outputs/exp001 \
    --epochs 3 \
    --batch_size 4
```

主要参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model_id` | `/home/share/models/Llama-3.2-1B` | 基座模型路径 |
| `--train_file` | `./output/train_round1.jsonl` | 训练集路径 |
| `--val_file` | `./output/val.jsonl` | 验证集路径 |
| `--output_dir` | `./outputs/experiments/exp001_multiturn_classifier` | 输出目录 |
| `--epochs` | `3` | 训练轮数 |
| `--batch_size` | `4` | 批大小 |
| `--max_length` | `1024` | 最大 token 长度 |
| `--no_lora` | — | 禁用 LoRA（全参数微调） |
| `--no_poe` | — | 禁用 Product of Experts 机制 |

训练产物保存在 `outputs/` 下：
- `best_model.pt` — 最佳验证准确率的完整权重
- `final_model/` — 最终模型（可直接用 `from_pretrained` 加载）
- `lora_adapter/` — LoRA 适配器
- `reports/` — 每 epoch 分类报告
- `training_metrics.png` — 损失与指标曲线
- `metrics_history.json` — 各 epoch 数值记录

### 3. 第二轮采样与训练

```bash
# 生成第二轮训练集（自动排除已用 ID）
python sample_round2.py

# 用第二轮数据训练
python train.py \
    --train_file output/train_round2.jsonl \
    --output_dir outputs/exp002
```

---

## 数据格式

每行一个 JSON（pool 文件无 `label` 字段）：

```json
{
  "id": "oasst2_<leaf_message_id>",
  "source": "oasst2",
  "original_id": "<原始 ID>",
  "label": 0,
  "conversations": [
    {"role": "user",      "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "user",      "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "metadata": {
    "leaf_message_id": "...",
    "tree_id": "...",
    "num_turns": 4,
    "synthetic": false,
    "rank": 0
  }
}
```

标签：`oasst2 → label=0`，`Magpie → label=1`

详细字段说明见 [read.md](read.md)。

---

## 模型架构

`ClassifierWithLoRAAndPoE`：基于 Causal LM + LoRA，最后一个非 padding token 的隐藏状态做分类头。

**Product of Experts (PoE)**：
- 主分类器：基于语义内容的 logits
- 长度专家：仅基于序列长度的简单 MLP
- 最终 logits = 主分类器 + 长度专家

PoE 的作用是迫使主模型学习真正的语义特征，而不是依赖文本长度等表浅特征。

---

## 多轮训练计划

| 轮次 | 脚本 | 训练集 | 条数 | 累计已用 |
|---|---|---|---|---|
| Round 1 | `extract_datasets.py` | `train_round1.jsonl` | 10,000 | 11,000 |
| Round 2 | `sample_round2.py` | `train_round2.jsonl` | 11,000 | 22,000 |
| Round 3+ | 修改 `sample_round2.py` 参数 | — | 自定义 | — |

oasst2 可用总量约 17,427 条（train split），验证集固定来自官方 validation split（916 条中取 500）。

---

## Git 工作流

```bash
# 本地修改完成后
git add <修改的文件>
git commit -m "描述"
git push

# 跳板机/服务器拉取
git pull
python train.py ...
```

首次在新机器上部署：
```bash
git clone https://github.com/Naoziyoupao/work2_new.git
cd work2_new
pip install datasets transformers peft torch tqdm scikit-learn matplotlib
python extract_datasets.py   # 重新生成 pool 和训练集
```
