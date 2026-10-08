r"""
filter_short_names.py — post-process an existing matching_results.tsv to
drop matches where either side's business_name is too short to be a
reliable string-similarity signal (e.g. "PS", "EC", "KC" style 2-letter
codes that can falsely score high similarity against unrelated names).

This does NOT retrain anything -- it just re-reads your already-generated
matching_results.tsv and the corresponding source files, and writes a
filtered version. Works against either a test/ or train/ split -- just
point --data-dir at the folder containing sourceN.tsv files (it
auto-detects the "test_" or "train_" filename prefix).

Usage (test set, no ground truth):
    python3 filter_short_names.py \
        --matching output_run\matching_results.tsv \
        --data-dir dataset_small\test \
        --out output_run\matching_results_filtered.tsv \
        --min-len 4

Usage (validation split, to score against ground truth):
    python3 filter_short_names.py \
        --matching output_run\val_matching_results.tsv \
        --data-dir dataset_small\train \
        --out output_run\val_matching_results_filtered.tsv \
        --min-len 4
"""

import argparse
import glob
import re
from pathlib import Path

import pandas as pd


def normalize_name(s) -> str:
    if pd.isna(s):
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def find_source_file(data_dir: Path, source_num: int) -> Path:
    matches = list(data_dir.glob(f"*_source{source_num}.tsv")) + list(data_dir.glob(f"source{source_num}.tsv"))
    if not matches:
        raise FileNotFoundError(f"No source{source_num}.tsv-style file found in {data_dir}")
    return matches[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", required=True)
    ap.add_argument("--data-dir", required=True,
                     help="folder containing sourceN.tsv files (test/ or train/, any name prefix)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-len", type=int, default=4,
                     help="names (post-normalization) shorter than this are treated as uninformative")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    s1 = pd.read_csv(find_source_file(data_dir, 1), sep="\t", dtype=str).fillna("")
    s2 = pd.read_csv(find_source_file(data_dir, 2), sep="\t", dtype=str).fillna("")
    s3 = pd.read_csv(find_source_file(data_dir, 3), sep="\t", dtype=str).fillna("")

    name_by_id = {}
    for df in (s1, s2, s3):
        for row in df.itertuples(index=False):
            name_by_id[row.entity_id] = normalize_name(row.business_name)

    def is_short(entity_id: str) -> bool:
        name = name_by_id.get(entity_id, "")
        return len(name) < args.min_len

    matching = pd.read_csv(args.matching, sep="\t", dtype=str).fillna("")

    dropped_count = 0
    kept_rows = []
    for row in matching.itertuples(index=False):
        s1_id = row.source1_entity_id
        ids_str = str(row.matched_entity_ids).strip()
        if not ids_str:
            kept_rows.append((s1_id, ""))
            continue

        ids = ids_str.split(",")
        kept_ids = []
        for mid in ids:
            if is_short(s1_id) or is_short(mid):
                dropped_count += 1
                print(f"  DROP: {s1_id} ({name_by_id.get(s1_id,'?')!r}) -- "
                      f"{mid} ({name_by_id.get(mid,'?')!r})")
            else:
                kept_ids.append(mid)
        kept_rows.append((s1_id, ",".join(kept_ids)))

    out_df = pd.DataFrame(kept_rows, columns=["source1_entity_id", "matched_entity_ids"])
    out_df.to_csv(args.out, sep="\t", index=False)
    print(f"\nDropped {dropped_count} matches involving names shorter than {args.min_len} chars.")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()