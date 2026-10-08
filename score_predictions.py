"""
score_predictions.py — compute the challenge's actual macro F_0.5 metric
by comparing a predictions file (e.g. val_matching_results.tsv, or a
filtered version of it) against train_ground_truth.tsv.

Only works where ground truth exists -- i.e. against the validation split
from train_eval mode, never against the real test set (which has no
ground truth by design). This is how you get a real, numeric answer to
"did this filter/fix actually help?" instead of eyeballing dropped rows.

Usage:
    python3 score_predictions.py \
        --predictions output_run\val_matching_results.tsv \
        --ground-truth dataset_small\train\train_ground_truth.tsv
"""

import argparse

import numpy as np
import pandas as pd


def macro_f_half(preds_by_s1: dict, gt_map: dict, all_s1_ids) -> float:
    scores = []
    for s1_id in all_s1_ids:
        pred = preds_by_s1.get(s1_id, set())
        true = gt_map.get(s1_id, set())
        if not pred and not true:
            scores.append(1.0)
            continue
        tp = len(pred & true)
        fp = len(pred - true)
        fn = len(true - pred)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        denom = 0.25 * precision + recall
        scores.append((1.25 * precision * recall) / denom if denom > 0 else 0.0)
    return float(np.mean(scores))


def load_id_map(path: str, id_col: str, ids_col: str) -> dict:
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    out = {}
    for row in df.itertuples(index=False):
        ids_str = str(getattr(row, ids_col)).strip()
        out[getattr(row, id_col)] = set(ids_str.split(",")) if ids_str else set()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True,
                     help="a matching_results-style tsv (source1_entity_id, matched_entity_ids)")
    ap.add_argument("--ground-truth", required=True,
                     help="train_ground_truth.tsv (must be from the SAME dataset the predictions came from)")
    args = ap.parse_args()

    preds = load_id_map(args.predictions, "source1_entity_id", "matched_entity_ids")
    gt = load_id_map(args.ground_truth, "source1_entity_id", "matched_entity_ids")

    # only score S1 entities that appear in the predictions file (e.g. the
    # validation split), not every entity in the full ground truth file
    all_s1_ids = list(preds.keys())
    missing_gt = [s1 for s1 in all_s1_ids if s1 not in gt]
    if missing_gt:
        print(f"WARNING: {len(missing_gt)} predicted S1 entities have no ground-truth row "
              f"(they'll be scored as if their true match set is empty) -- e.g. {missing_gt[:5]}")

    score = macro_f_half(preds, gt, all_s1_ids)
    n_pred_nonempty = sum(1 for v in preds.values() if v)
    n_gt_nonempty = sum(1 for s1 in all_s1_ids if gt.get(s1))

    print(f"Scored {len(all_s1_ids)} S1 entities")
    print(f"  predictions: {n_pred_nonempty} non-empty, {len(all_s1_ids)-n_pred_nonempty} empty")
    print(f"  ground truth: {n_gt_nonempty} non-empty, {len(all_s1_ids)-n_gt_nonempty} empty")
    print(f"Macro F_0.5: {score:.4f}")


if __name__ == "__main__":
    main()