"""
Round 2 sampling script.

Reads from existing pool files (output/pool_*.jsonl), excludes all IDs
already used in round 1 (train + val), then samples:
  - 5500 from oasst2 train pool  (label=0)
  - 5500 from Magpie pool        (label=1)
  = 11000 total training records

Validation set is unchanged (val.jsonl stays the same across rounds).
Appends round 2 IDs into selected_ids.json for future exclusion.
"""

import json
import random
import os

SEED        = 43                   # different seed for round 2
TRAIN_N     = 5500                 # per source
OUTPUT_DIR  = "output"
POOL_OASST2 = os.path.join(OUTPUT_DIR, "pool_oasst2_train.jsonl")
POOL_MAGPIE = os.path.join(OUTPUT_DIR, "pool_magpie.jsonl")
SELECTED_IDS_FILE = os.path.join(OUTPUT_DIR, "selected_ids.json")
OUT_TRAIN   = os.path.join(OUTPUT_DIR, "train_round2.jsonl")

random.seed(SEED)


def load_jsonl(path):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(data)} records → {path}")


def main():
    # 1. Load already-used IDs (round 1 train + val)
    print("=== Loading used IDs from round 1 ===")
    with open(SELECTED_IDS_FILE, encoding="utf-8") as f:
        selected = json.load(f)

    used_ids = set(selected.get("train_round1", []))
    used_ids |= set(selected.get("val", []))
    if "train_round2" in selected:
        used_ids |= set(selected["train_round2"])
    print(f"  Already used: {len(used_ids)} IDs")

    # 2. Load pools and filter out used IDs
    print("\n=== Loading oasst2 train pool ===")
    oasst2_pool = load_jsonl(POOL_OASST2)
    oasst2_avail = [r for r in oasst2_pool if r["id"] not in used_ids]
    print(f"  Pool: {len(oasst2_pool)}, available after exclusion: {len(oasst2_avail)}")

    print("\n=== Loading Magpie pool ===")
    magpie_pool = load_jsonl(POOL_MAGPIE)
    magpie_avail = [r for r in magpie_pool if r["id"] not in used_ids]
    print(f"  Pool: {len(magpie_pool)}, available after exclusion: {len(magpie_avail)}")

    # 3. Check availability
    if len(oasst2_avail) < TRAIN_N:
        raise ValueError(
            f"oasst2 available ({len(oasst2_avail)}) < requested ({TRAIN_N})"
        )
    if len(magpie_avail) < TRAIN_N:
        raise ValueError(
            f"Magpie available ({len(magpie_avail)}) < requested ({TRAIN_N})"
        )

    # 4. Sample
    print(f"\n=== Sampling {TRAIN_N} from each source (seed={SEED}) ===")
    train_oasst2 = random.sample(oasst2_avail, TRAIN_N)
    train_magpie = random.sample(magpie_avail, TRAIN_N)

    # Ensure labels are set
    for rec in train_oasst2:
        rec["label"] = 0
    for rec in train_magpie:
        rec["label"] = 1

    train_data = train_oasst2 + train_magpie
    random.shuffle(train_data)

    print(f"  oasst2 sampled: {len(train_oasst2)}")
    print(f"  Magpie sampled: {len(train_magpie)}")
    print(f"  Total:          {len(train_data)}")

    # 5. Save train_round2.jsonl
    print("\n=== Saving ===")
    save_jsonl(train_data, OUT_TRAIN)

    # 6. Update selected_ids.json with round 2 IDs
    selected["train_round2"] = [item["id"] for item in train_data]
    selected["round2_stats"] = {
        "seed": SEED,
        "train_size_per_source": TRAIN_N,
        "train_from_oasst2": len(train_oasst2),
        "train_from_magpie": len(train_magpie),
        "oasst2_pool_remaining_after_r2": len(oasst2_avail) - TRAIN_N,
        "magpie_pool_remaining_after_r2": len(magpie_avail) - TRAIN_N,
    }
    with open(SELECTED_IDS_FILE, "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)
    print(f"  Updated selected IDs → {SELECTED_IDS_FILE}")

    # 7. Summary
    print("\n=== Summary ===")
    print(f"  train_round2.jsonl : {len(train_data)} records")
    print(f"    oasst2 (label=0) : {len(train_oasst2)}")
    print(f"    Magpie  (label=1): {len(train_magpie)}")
    print(f"  val.jsonl unchanged (500 oasst2 + 500 Magpie from round 1)")
    print(f"  oasst2 remaining for future rounds: {len(oasst2_avail) - TRAIN_N}")
    print(f"  Magpie  remaining for future rounds: {len(magpie_avail) - TRAIN_N}")
    print("\nDone!")


if __name__ == "__main__":
    main()
