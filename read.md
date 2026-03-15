# 多轮对话训练数据集说明

## 数据来源

| 来源 | HuggingFace 地址 | 类型 |
|---|---|---|
| oasst2 | `OpenAssistant/oasst2` | 人类标注多轮对话 |
| Magpie | `Magpie-Align/Magpie-Pro-MT-300K-v0.1` | Llama-3-70B 合成多轮对话 |

---

## 文件列表（`output/` 目录）

| 文件 | 条数 | 大小 | 说明 |
|---|---|---|---|
| `pool_oasst2.jsonl` | 18,343 | 44.5 MB | oasst2 全量英文多轮数据（无 label 字段） |
| `pool_magpie.jsonl` | 300,000 | 1,687.8 MB | Magpie 全量多轮数据（无 label 字段） |
| `train_round1.jsonl` | 10,000 | 40.5 MB | **第一轮训练集**（含 label） |
| `val.jsonl` | 1,000 | 4.0 MB | **验证集**（含 label） |
| `selected_ids.json` | — | 0.5 MB | 已用 ID 记录，第二轮采样时排除 |

---

## 采样方案

- 随机种子：`seed = 42`
- 多轮定义：`conversations` 至少 4 个 turn（≥ 2 轮 user-assistant 交互）
- 训练集：从 oasst2 和 Magpie **各独立随机抽取 5000 条**，共 10,000 条
- 验证集：从各自剩余数据中**各取 500 条**，共 1,000 条
- 训练集与验证集之间**无重叠**
- 标签：`oasst2 → label=0`，`Magpie → label=1`

---

## 统一数据格式

训练集和验证集每行一个 JSON 对象：

```json
{
  "id":          "oasst2_<leaf_message_id> | magpie_<uuid>",
  "source":      "oasst2 | magpie",
  "original_id": "<原始数据集中的 ID>",
  "label":       0,
  "conversations": [
    {"role": "user",      "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "user",      "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "metadata": { ... }
}
```

Pool 文件格式相同，但**没有** `label` 字段。

### metadata 字段说明

**oasst2：**

| 字段 | 说明 |
|---|---|
| `leaf_message_id` | 对话最后一条消息的 ID |
| `tree_id` | 对话树根节点 ID（即第一条 user 消息的 ID） |
| `num_turns` | 对话总 turn 数 |
| `synthetic` | 是否为合成数据（bool） |
| `rank` | 该叶子节点在同级中的排名（0 = 最优） |

**Magpie：**

| 字段 | 说明 |
|---|---|
| `model` | 生成模型名称（`meta-llama/Meta-Llama-3-70B-Instruct`） |
| `gen_input_config` | 生成参数（temperature、top_p 等） |
| `num_turns` | 对话总 turn 数 |

---

## Turn 数分布（train_round1.jsonl）

| turn 数 | 条数 |
|---|---|
| 4（2轮交互） | 9,706 |
| 6（3轮交互） | 294 |

---

## 读取示例

```python
import json

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]

train = load_jsonl("output/train_round1.jsonl")
val   = load_jsonl("output/val.jsonl")

# 访问字段
sample = train[0]
print(sample["source"])          # "oasst2" 或 "magpie"
print(sample["label"])           # 0 或 1
print(sample["conversations"])   # list of {"role": ..., "content": ...}
```

---

## 第二轮数据采样

`selected_ids.json` 记录了第一轮所有已使用的 ID（训练集 + 验证集，共 11,000 个）。
第二轮采样时从 pool 文件中排除这些 ID：

```python
import json

with open("output/selected_ids.json", encoding="utf-8") as f:
    used = json.load(f)

used_ids = set(used["train_round1"] + used["val"])

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]

pool_oasst2 = [x for x in load_jsonl("output/pool_oasst2.jsonl") if x["id"] not in used_ids]
pool_magpie = [x for x in load_jsonl("output/pool_magpie.jsonl") if x["id"] not in used_ids]

# pool_oasst2: 18343 - 5000(train) - 500(val) = 12843 条可用
# pool_magpie: 300000 - 5000(train) - 500(val) = 294500 条可用
```

---

## 生成脚本

| 脚本 | 用途 |
|---|---|
| `extract_datasets.py` | 完整流程：从 HuggingFace 下载 → 提取 → 采样 → 保存 |
| `resample.py` | 仅重新采样：读取已有 pool 文件 → 调整采样数量 → 重新生成 train/val |
