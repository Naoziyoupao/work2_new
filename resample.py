"""
Resample training and validation sets from existing pool files.

Changes vs. original sampling:
  - Labels added: oasst2 -> label=0, Magpie -> label=1
  - Training set: sample TRAIN_SIZE_PER_SOURCE from EACH source separately
    (2500 oasst2 + 2500 Magpie = 5000 total)
  - Validation set: sample VAL_SIZE_PER_SOURCE from each remaining
    (500 oasst2 + 500 Magpie = 1000 total)
  - Records must not overlap between train and val
"""

import json
import random
import os

SEED = 42
TRAIN_SIZE_PER_SOURCE = 5000
VAL_SIZE_PER_SOURCE   = 500
OUTPUT_DIR = "output"

LABEL_MAP = {"oasst2": 0, "magpie": 1}

random.seed(SEED)


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"  Loaded {len(data)} records from {path}")
    return data


def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(data)} records -> {path}")


def add_label(records):
    """Add label field in-place based on source."""
    for rec in records:
        rec["label"] = LABEL_MAP[rec["source"]]
    return records


def sample_from_source(data, train_n, val_n, source_name):
    total = len(data)
    need = train_n + val_n
    if total < need:
        raise ValueError(
            f"{source_name}: pool size ({total}) < train+val need ({need}). "
            f"Reduce sizes or use fewer samples."
        )

    indices = list(range(total))
    random.shuffle(indices)

    train_idx = indices[:train_n]
    val_idx   = indices[train_n : train_n + val_n]

    train = [data[i] for i in train_idx]
    val   = [data[i] for i in val_idx]
    return train, val


if __name__ == "__main__":
    print("=== Loading pool files ===")
    oasst2_data = load_jsonl(os.path.join(OUTPUT_DIR, "pool_oasst2.jsonl"))
    magpie_data = load_jsonl(os.path.join(OUTPUT_DIR, "pool_magpie.jsonl"))

    print("\n=== Adding labels ===")
    add_label(oasst2_data)  # label=0
    add_label(magpie_data)  # label=1
    print(f"  oasst2: label=0 ({len(oasst2_data)} records)")
    print(f"  Magpie: label=1 ({len(magpie_data)} records)")

    print("\n=== Sampling (per-source) ===")
    train_oasst2, val_oasst2 = sample_from_source(
        oasst2_data, TRAIN_SIZE_PER_SOURCE, VAL_SIZE_PER_SOURCE, "oasst2"
    )
    train_magpie, val_magpie = sample_from_source(
        magpie_data, TRAIN_SIZE_PER_SOURCE, VAL_SIZE_PER_SOURCE, "Magpie"
    )

    train_data = train_oasst2 + train_magpie
    val_data   = val_oasst2   + val_magpie

    # Shuffle combined train/val so sources are interleaved
    random.shuffle(train_data)
    random.shuffle(val_data)

    print(f"  Train total: {len(train_data)} "
          f"(oasst2={len(train_oasst2)}, magpie={len(train_magpie)})")
    print(f"  Val   total: {len(val_data)} "
          f"(oasst2={len(val_oasst2)}, magpie={len(val_magpie)})")

    print("\n=== Saving outputs ===")
    save_jsonl(train_data, os.path.join(OUTPUT_DIR, "train_round1.jsonl"))
    save_jsonl(val_data,   os.path.join(OUTPUT_DIR, "val.jsonl"))

    # Record selected IDs for round-2 exclusion
    selected_ids = {
        "train_round1": [item["id"] for item in train_data],
        "val":          [item["id"] for item in val_data],
        "seed": SEED,
        "train_size_per_source": TRAIN_SIZE_PER_SOURCE,
        "val_size_per_source": VAL_SIZE_PER_SOURCE,
        "label_map": LABEL_MAP,
        "stats": {
            "pool_oasst2_total": len(oasst2_data),
            "pool_magpie_total": len(magpie_data),
            "train_from_oasst2": len(train_oasst2),
            "train_from_magpie": len(train_magpie),
            "val_from_oasst2":   len(val_oasst2),
            "val_from_magpie":   len(val_magpie),
        },
    }

    ids_path = os.path.join(OUTPUT_DIR, "selected_ids.json")
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(selected_ids, f, ensure_ascii=False, indent=2)
    print(f"  Saved selected IDs -> {ids_path}")

    print("\n=== Done ===")
    print(f"  train_round1.jsonl : {len(train_data)} samples "
          f"(oasst2 label=0 x{len(train_oasst2)}, magpie label=1 x{len(train_magpie)})")
    print(f"  val.jsonl          : {len(val_data)} samples "
          f"(oasst2 label=0 x{len(val_oasst2)}, magpie label=1 x{len(val_magpie)})")
    print(f"  selected_ids.json  : {len(selected_ids['train_round1'])+len(selected_ids['val'])} IDs recorded")
