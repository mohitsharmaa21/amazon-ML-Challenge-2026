"""
subsample_dataset.py — shrink the ~1GB challenge dataset to a manageable
size for fast local iteration, WITHOUT breaking ground-truth referential
integrity (every true match for a sampled Source-1 entity is guaranteed
to still be present in the sampled Source-2/3 files).

Usage:
    python3 subsample_dataset.py \
        --in-dir  dataset \
        --out-dir dataset_small \
        --n-train-s1 2000 \
        --n-test-s1 500 \
        --noise-ratio 3.0
"""

import argparse
import random
from pathlib import Path

import pandas as pd

random.seed(42)


def load_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str).fillna("")


def write_tsv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)
    print(f"  wrote {path}  ({len(df):,} rows, {path.stat().st_size/1e6:.1f} MB)")


def subsample_train(in_dir: Path, out_dir: Path, n_s1: int, noise_ratio: float):
    print("Loading full training files (this may take a bit for 1GB)...")
    s1 = load_tsv(in_dir / "train" / "train_source1.tsv")
    s2 = load_tsv(in_dir / "train" / "train_source2.tsv")
    s3 = load_tsv(in_dir / "train" / "train_source3.tsv")
    gt = load_tsv(in_dir / "train" / "train_ground_truth.tsv")

    n_s1 = min(n_s1, len(s1))
    s1_sample = s1.sample(n=n_s1, random_state=42)
    sampled_s1_ids = set(s1_sample["entity_id"])

    gt_sample = gt[gt["source1_entity_id"].isin(sampled_s1_ids)].copy()

    required_ids = set()
    for ids_str in gt_sample["matched_entity_ids"]:
        ids_str = str(ids_str).strip()
        if ids_str:
            required_ids.update(ids_str.split(","))

    required_s2_ids = {i for i in required_ids if i.startswith("S2-")}
    required_s3_ids = {i for i in required_ids if i.startswith("S3-")}

    n_noise = int(n_s1 * noise_ratio)
    s2_noise_pool = s2[~s2["entity_id"].isin(required_s2_ids)]
    s3_noise_pool = s3[~s3["entity_id"].isin(required_s3_ids)]
    s2_noise = s2_noise_pool.sample(n=min(n_noise, len(s2_noise_pool)), random_state=42)
    s3_noise = s3_noise_pool.sample(n=min(n_noise, len(s3_noise_pool)), random_state=43)

    s2_final = pd.concat([s2[s2["entity_id"].isin(required_s2_ids)], s2_noise]).drop_duplicates("entity_id")
    s3_final = pd.concat([s3[s3["entity_id"].isin(required_s3_ids)], s3_noise]).drop_duplicates("entity_id")

    print(f"Train sample: {len(s1_sample)} S1 entities, "
          f"{len(s2_final)} S2 records ({len(required_s2_ids)} required matches), "
          f"{len(s3_final)} S3 records ({len(required_s3_ids)} required matches)")

    write_tsv(s1_sample, out_dir / "train" / "train_source1.tsv")
    write_tsv(s2_final, out_dir / "train" / "train_source2.tsv")
    write_tsv(s3_final, out_dir / "train" / "train_source3.tsv")
    write_tsv(gt_sample, out_dir / "train" / "train_ground_truth.tsv")


def subsample_test(in_dir: Path, out_dir: Path, n_s1: int):
    test_dir = in_dir / "test"
    if not test_dir.exists():
        print("No test/ dir found, skipping test subsampling.")
        return

    print("Loading full test files...")
    s1 = load_tsv(test_dir / "test_source1.tsv")
    s2 = load_tsv(test_dir / "test_source2.tsv")
    s3 = load_tsv(test_dir / "test_source3.tsv")

    n_s1 = min(n_s1, len(s1))
    s1_sample = s1.sample(n=n_s1, random_state=42)

    frac = n_s1 / len(s1)
    s2_sample = s2.sample(frac=min(frac * 2, 1.0), random_state=42)
    s3_sample = s3.sample(frac=min(frac * 2, 1.0), random_state=43)

    print(f"Test sample: {len(s1_sample)} S1, {len(s2_sample)} S2, {len(s3_sample)} S3 "
          "(NOTE: no ground truth exists for test, so this is for speed-testing your "
          "pipeline only -- your real leaderboard submission must run on the FULL test set)")

    write_tsv(s1_sample, out_dir / "test" / "test_source1.tsv")
    write_tsv(s2_sample, out_dir / "test" / "test_source2.tsv")
    write_tsv(s3_sample, out_dir / "test" / "test_source3.tsv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="full dataset dir (contains train/ and test/)")
    ap.add_argument("--out-dir", required=True, help="where to write the smaller dataset")
    ap.add_argument("--n-train-s1", type=int, default=2000, help="# Source-1 training entities to keep")
    ap.add_argument("--n-test-s1", type=int, default=500, help="# Source-1 test entities to keep")
    ap.add_argument("--noise-ratio", type=float, default=3.0,
                     help="extra non-matching S2/S3 records per sampled S1 entity, "
                          "to keep blocking difficulty realistic")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    print("=== Subsampling TRAIN ===")
    subsample_train(in_dir, out_dir, args.n_train_s1, args.noise_ratio)

    print("\n=== Subsampling TEST ===")
    subsample_test(in_dir, out_dir, args.n_test_s1)

    print(f"\nDone. Point --data-dir at {out_dir} for fast local iteration.")
    print("Remember: your FINAL leaderboard submission must be generated from the FULL, unmodified test set.")


if __name__ == "__main__":
    main()