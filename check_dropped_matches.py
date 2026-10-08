"""
check_dropped_matches.py — check a specific list of (source1_id, matched_id)
pairs against ground truth, to see whether they were real matches (that a
filter incorrectly removed) or genuine false positives (correctly removed).

Usage:
    python3 check_dropped_matches.py --ground-truth dataset_small\train\train_ground_truth.tsv
"""

import argparse
import pandas as pd

DROPPED_PAIRS = [
    ("S1-377014380", "S2-266197882"),
    ("S1-149615573", "S3-490096245"),
    ("S1-717223466", "S3-60044945"),
    ("S1-717223466", "S3-950690694"),
    ("S1-343226247", "S3-538597075"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ground-truth", required=True)
    args = ap.parse_args()

    gt = pd.read_csv(args.ground_truth, sep="\t", dtype=str).fillna("")
    gt_map = {}
    for row in gt.itertuples(index=False):
        ids_str = str(row.matched_entity_ids).strip()
        gt_map[row.source1_entity_id] = set(ids_str.split(",")) if ids_str else set()

    true_matches_lost = 0
    false_positives_removed = 0
    for s1, matched in DROPPED_PAIRS:
        was_true = matched in gt_map.get(s1, set())
        if was_true:
            true_matches_lost += 1
            verdict = "TRUE MATCH -- we lost a real match!"
        else:
            false_positives_removed += 1
            verdict = "false positive -- correctly removed"
        print(f"{s1} -- {matched}: {verdict}")

    print(f"\nSummary: {true_matches_lost} true matches lost, {false_positives_removed} false positives correctly removed")


if __name__ == "__main__":
    main()