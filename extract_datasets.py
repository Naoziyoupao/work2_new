"""
Dataset extraction script for multi-turn training data.
Sources:
  - OpenAssistant/oasst2 (English multi-turn conversations)
  - Magpie-Align/Magpie-Pro-MT-300K-v0.1 (English multi-turn conversations)

Output format (unified):
{
    "id": str,            # unique ID: "{source}_{original_id}"
    "source": str,        # "oasst2" or "magpie"
    "original_id": str,   # original identifier in the source dataset
    "conversations": [    # list of turns, alternating user/assistant
        {"role": "user", "content": str},
        {"role": "assistant", "content": str},
        ...
    ],
    "metadata": dict      # source-specific extra fields
}

Sampling strategy:
  - Training (round 1): 5000 from combined pool (sampled uniformly)
  - Validation: 500 from oasst2 remaining + 500 from Magpie remaining
  - Selected IDs recorded in selected_ids.json for round 2 exclusion
"""

import json
import random
import os
from collections import defaultdict
from datasets import load_dataset
from tqdm import tqdm

SEED = 42
TRAIN_SIZE_PER_SOURCE = 5000   # sampled independently from each source
VAL_SIZE_PER_SOURCE   = 500    # sampled independently from each source
OUTPUT_DIR = "output"

# Labels: oasst2 -> 0, Magpie -> 1
LABEL_MAP = {"oasst2": 0, "magpie": 1}

random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 1. Extract oasst2 English multi-turn data
# ─────────────────────────────────────────────

def build_oasst2_conversations(dataset_split):
    """
    Reconstruct conversation threads from oasst2 message tree.
    Returns list of conversation dicts in unified format.

    oasst2 schema fields used:
      message_id, parent_id, role (prompter/assistant),
      text, lang, rank, deleted, synthetic
    """
    # Index messages by message_id
    messages = {}
    for row in tqdm(dataset_split, desc="Indexing oasst2 messages"):
        messages[row["message_id"]] = row

    # Build children map
    children = defaultdict(list)
    for msg_id, msg in messages.items():
        if msg["parent_id"] is not None:
            children[msg["parent_id"]].append(msg_id)

    # Sort children by rank (ascending; rank=0 is best)
    for parent_id in children:
        children[parent_id].sort(
            key=lambda mid: messages[mid].get("rank") if messages[mid].get("rank") is not None else 999
        )

    # Find root messages (no parent)
    roots = [mid for mid, msg in messages.items() if msg["parent_id"] is None]

    conversations = []

    def traverse(msg_id, path):
        """DFS: collect all root-to-leaf paths, keep only valid English threads."""
        msg = messages[msg_id]

        # Skip deleted messages
        if msg.get("deleted", False):
            return

        # Only keep English messages (check on leaf assistant turns)
        path = path + [msg_id]
        child_ids = children.get(msg_id, [])

        if not child_ids:
            # Leaf node — reconstruct conversation from path
            thread = [messages[mid] for mid in path]

            # Check: all messages in thread should be English
            langs = set(m.get("lang", "") for m in thread)
            if not (langs <= {"en", None, ""}):
                return

            # Check: must start with prompter and alternate properly
            # oasst2 roles: "prompter" (user) and "assistant"
            roles = [m["role"] for m in thread]
            if roles[0] != "prompter":
                return

            # Build conversation turns
            conv_turns = []
            for m in thread:
                role = "user" if m["role"] == "prompter" else "assistant"
                text = (m.get("text") or "").strip()
                if not text:
                    return  # skip empty messages
                conv_turns.append({"role": role, "content": text})

            # Multi-turn: at least 2 user+assistant exchanges (4 turns)
            if len(conv_turns) < 4:
                return

            # Must end with assistant
            if conv_turns[-1]["role"] != "assistant":
                return

            # Build metadata from leaf message
            leaf = thread[-1]
            meta = {
                "leaf_message_id": leaf["message_id"],
                "tree_id": thread[0]["message_id"],
                "num_turns": len(conv_turns),
                "synthetic": leaf.get("synthetic"),
                "rank": leaf.get("rank"),
            }

            conversations.append({
                "id": f"oasst2_{leaf['message_id']}",
                "source": "oasst2",
                "original_id": leaf["message_id"],
                "conversations": conv_turns,
                "metadata": meta,
            })
        else:
            for child_id in child_ids:
                traverse(child_id, path)

    for root_id in tqdm(roots, desc="Building oasst2 conversation threads"):
        traverse(root_id, [])

    return conversations


def extract_oasst2():
    """
    Load oasst2 train and validation splits separately to avoid cross-split
    content overlap (tree-structured data shares common prefixes).
      - train split  → training pool
      - validation split → validation pool (used directly, not sampled from train remainder)
    """
    print("\n=== Loading oasst2 (train split) ===")
    ds_train = load_dataset("OpenAssistant/oasst2", split="train")
    train_convs = build_oasst2_conversations(ds_train)
    print(f"  oasst2 train: extracted {len(train_convs)} English multi-turn conversations")

    print("\n=== Loading oasst2 (validation split) ===")
    ds_val = load_dataset("OpenAssistant/oasst2", split="validation")
    val_convs = build_oasst2_conversations(ds_val)
    print(f"  oasst2 validation: extracted {len(val_convs)} English multi-turn conversations")

    return train_convs, val_convs


# ─────────────────────────────────────────────
# 2. Extract Magpie English multi-turn data
# ─────────────────────────────────────────────

def extract_magpie():
    """
    Magpie-Pro-MT-300K-v0.1 schema:
      - uuid:           str  (unique ID)
      - model:          str  (generation model)
      - gen_input_config: dict
      - input1/output1, input2/output2: str (individual turn content)
      - conversations:  list of {"from": "human"/"gpt", "value": str}

    All data is English (generated by Llama 3). No language filtering needed.
    Multi-turn = conversations with >= 4 turns (>= 2 exchanges).
    """
    print("\n=== Loading Magpie-Pro-MT-300K-v0.1 ===")
    ds = load_dataset("Magpie-Align/Magpie-Pro-MT-300K-v0.1", split="train")

    print(f"  Total rows: {len(ds)}")
    print(f"  Columns: {ds.column_names}")

    conversations = []
    skipped = 0

    for idx, row in enumerate(tqdm(ds, desc="Processing Magpie")):
        raw_conv = row.get("conversations") or []

        if not raw_conv:
            skipped += 1
            continue

        conv_turns = []
        valid = True
        for turn in raw_conv:
            # Magpie uses {"from": "human"/"gpt", "value": "..."}
            from_role = (turn.get("from") or turn.get("role") or "").lower()
            content = (turn.get("value") or turn.get("content") or "").strip()

            if from_role in ("human", "user"):
                role = "user"
            elif from_role in ("gpt", "assistant", "bot"):
                role = "assistant"
            else:
                valid = False
                break

            if not content:
                valid = False
                break

            conv_turns.append({"role": role, "content": content})

        if not valid:
            skipped += 1
            continue

        # Multi-turn: at least 4 turns (2 user+assistant exchanges)
        if len(conv_turns) < 4:
            skipped += 1
            continue

        # Must start with user and end with assistant
        if conv_turns[0]["role"] != "user" or conv_turns[-1]["role"] != "assistant":
            skipped += 1
            continue

        original_id = str(row.get("uuid") or idx)

        # Metadata: keep scalar fields, exclude raw conversation list
        meta = {
            "model": row.get("model"),
            "gen_input_config": row.get("gen_input_config"),
            "num_turns": len(conv_turns),
        }

        conversations.append({
            "id": f"magpie_{original_id}",
            "source": "magpie",
            "original_id": original_id,
            "conversations": conv_turns,
            "metadata": meta,
        })

    print(f"  Magpie: extracted {len(conversations)} English multi-turn conversations (skipped {skipped})")
    return conversations


# ─────────────────────────────────────────────
# 3. Sample training and validation sets
# ─────────────────────────────────────────────

def _sample_one_source(data, train_n, val_n, name):
    """Shuffle data and split into train/val without overlap."""
    total = len(data)
    need = train_n + val_n
    if total < need:
        raise ValueError(
            f"{name}: pool size ({total}) < train+val need ({need})."
        )
    indices = list(range(total))
    random.shuffle(indices)
    train = [data[i] for i in indices[:train_n]]
    val   = [data[i] for i in indices[train_n : train_n + val_n]]
    return train, val


def sample_datasets(oasst2_train_data, oasst2_val_data, magpie_data):
    """
    Strategy:
      - Add label field: oasst2->0, Magpie->1
      - oasst2 training: sample TRAIN_SIZE_PER_SOURCE from train split
      - oasst2 validation: sample VAL_SIZE_PER_SOURCE from official validation split
        (avoids cross-split content overlap due to shared conversation prefixes)
      - Magpie: sample TRAIN_SIZE_PER_SOURCE + VAL_SIZE_PER_SOURCE independently
    """
    print(f"\n=== Sampling ===")
    print(f"  oasst2 train pool: {len(oasst2_train_data)}")
    print(f"  oasst2 val pool:   {len(oasst2_val_data)}")
    print(f"  Magpie pool:       {len(magpie_data)}")

    # Add labels
    for rec in oasst2_train_data:
        rec["label"] = LABEL_MAP["oasst2"]
    for rec in oasst2_val_data:
        rec["label"] = LABEL_MAP["oasst2"]
    for rec in magpie_data:
        rec["label"] = LABEL_MAP["magpie"]

    # oasst2: sample train from train split, val from validation split
    if len(oasst2_train_data) < TRAIN_SIZE_PER_SOURCE:
        raise ValueError(f"oasst2 train pool ({len(oasst2_train_data)}) < {TRAIN_SIZE_PER_SOURCE}")
    if len(oasst2_val_data) < VAL_SIZE_PER_SOURCE:
        raise ValueError(f"oasst2 val pool ({len(oasst2_val_data)}) < {VAL_SIZE_PER_SOURCE}")

    train_idx = list(range(len(oasst2_train_data)))
    random.shuffle(train_idx)
    train_oasst2 = [oasst2_train_data[i] for i in train_idx[:TRAIN_SIZE_PER_SOURCE]]

    val_idx = list(range(len(oasst2_val_data)))
    random.shuffle(val_idx)
    val_oasst2 = [oasst2_val_data[i] for i in val_idx[:VAL_SIZE_PER_SOURCE]]

    # Magpie: sample train + val from single pool
    train_magpie, val_magpie = _sample_one_source(
        magpie_data, TRAIN_SIZE_PER_SOURCE, VAL_SIZE_PER_SOURCE, "Magpie"
    )

    train_data = train_oasst2 + train_magpie
    val_data   = val_oasst2   + val_magpie

    random.shuffle(train_data)
    random.shuffle(val_data)

    print(f"  Train: {len(train_data)} (oasst2={len(train_oasst2)}, magpie={len(train_magpie)})")
    print(f"  Val:   {len(val_data)}   (oasst2={len(val_oasst2)}, magpie={len(val_magpie)})")

    return train_data, val_data


# ─────────────────────────────────────────────
# 4. Save outputs
# ─────────────────────────────────────────────

def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(data)} records → {path}")


def save_outputs(oasst2_train_data, oasst2_val_data, magpie_data, train_data, val_data):
    # Save complete extracted pools
    save_jsonl(oasst2_train_data, os.path.join(OUTPUT_DIR, "pool_oasst2_train.jsonl"))
    save_jsonl(oasst2_val_data,   os.path.join(OUTPUT_DIR, "pool_oasst2_val.jsonl"))
    save_jsonl(magpie_data,       os.path.join(OUTPUT_DIR, "pool_magpie.jsonl"))

    # Save training and validation sets
    save_jsonl(train_data, os.path.join(OUTPUT_DIR, "train_round1.jsonl"))
    save_jsonl(val_data,   os.path.join(OUTPUT_DIR, "val.jsonl"))

    # Save selected IDs for round 2 exclusion
    selected_ids = {
        "train_round1": [item["id"] for item in train_data],
        "val":          [item["id"] for item in val_data],
        "seed": SEED,
        "train_size_per_source": TRAIN_SIZE_PER_SOURCE,
        "val_size_per_source": VAL_SIZE_PER_SOURCE,
        "label_map": LABEL_MAP,
        "stats": {
            "pool_oasst2_train_total": len(oasst2_train_data),
            "pool_oasst2_val_total":   len(oasst2_val_data),
            "pool_magpie_total":       len(magpie_data),
            "train_from_oasst2": sum(1 for x in train_data if x["source"] == "oasst2"),
            "train_from_magpie": sum(1 for x in train_data if x["source"] == "magpie"),
            "val_from_oasst2":   sum(1 for x in val_data   if x["source"] == "oasst2"),
            "val_from_magpie":   sum(1 for x in val_data   if x["source"] == "magpie"),
        }
    }

    ids_path = os.path.join(OUTPUT_DIR, "selected_ids.json")
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(selected_ids, f, ensure_ascii=False, indent=2)
    print(f"  Saved selected IDs → {ids_path}")

    # Print summary
    print("\n=== Summary ===")
    s = selected_ids["stats"]
    print(f"  Pool — oasst2 train: {s['pool_oasst2_train_total']}, oasst2 val: {s['pool_oasst2_val_total']}, Magpie: {s['pool_magpie_total']}")
    total_train = s['train_from_oasst2'] + s['train_from_magpie']
    print(f"  Train round 1 ({total_train} total): oasst2={s['train_from_oasst2']} (label=0), magpie={s['train_from_magpie']} (label=1)")
    print(f"  Val ({VAL_SIZE_PER_SOURCE} each):   oasst2={s['val_from_oasst2']} [from official val split], magpie={s['val_from_magpie']}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    oasst2_train_data, oasst2_val_data = extract_oasst2()
    magpie_data = extract_magpie()

    train_data, val_data = sample_datasets(oasst2_train_data, oasst2_val_data, magpie_data)

    print("\n=== Saving outputs ===")
    save_outputs(oasst2_train_data, oasst2_val_data, magpie_data, train_data, val_data)

    print("\nDone! Output files in ./output/")
    print("  pool_oasst2_train.jsonl — oasst2 train split multi-turn EN conversations")
    print("  pool_oasst2_val.jsonl   — oasst2 validation split multi-turn EN conversations")
    print("  pool_magpie.jsonl       — all extracted Magpie multi-turn EN conversations")
    print("  train_round1.jsonl      — 5000+5000 training samples (round 1)")
    print("  val.jsonl               — 500+500 validation samples")
    print("  selected_ids.json       — IDs to exclude in round 2")
