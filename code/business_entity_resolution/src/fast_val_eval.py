import io, pickle, random
from pathlib import Path
import numpy as np
import pandas as pd
import entilink
from collections import defaultdict
from er_entilink_pipeline import load_source, df_to_entilink_dict, apply_reuse_cap, macro_f_half, report_dropped_pairs, apply_min_name_len_filter

# We want the same val_ids as before. The previous run set random.seed(42) before splitting.
random.seed(42)

data_dir = Path("../../../dataset_small")
out_dir = Path("../../../output_run")

s1 = load_source(data_dir / "train" / "train_source1.tsv")
s2 = load_source(data_dir / "train" / "train_source2.tsv")
s3 = load_source(data_dir / "train" / "train_source3.tsv")
gt = pd.read_csv(data_dir / "train" / "train_ground_truth.tsv", sep="\t", dtype=str).fillna("")

s1_ids = s1["entity_id"].tolist()
random.shuffle(s1_ids)
cut = int(len(s1_ids) * 0.8)
val_ids = set(s1_ids[cut:])

s1_val = s1[s1["entity_id"].isin(val_ids)]
data_1_val = df_to_entilink_dict(s1_val)
data_2 = df_to_entilink_dict(pd.concat([s2, s3], ignore_index=True))

gt_val = gt[gt["source1_entity_id"].isin(val_ids)]
gt_map = {}
for row in gt_val.itertuples(index=False):
    ids = str(row.matched_entity_ids).strip()
    gt_map[row.source1_entity_id] = set(ids.split(",")) if ids else set()

val_s1_ids = s1_val["entity_id"].tolist()

print("Loading existing entilink settings...")
with open(out_dir / "entilink_settings.bin", "rb") as f:
    linker = entilink.StaticRecordLink(f, num_cores=1)

print("Scoring validation pairs...")
pairs = linker.pairs(data_1_val, data_2)
pair_scores = linker.score(pairs)

# BASELINE
print("\n--- BASELINE ---")
def tune_threshold(pair_scores, gt_map, all_s1_ids, apply_cap=False):
    best_t, best_score = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.05):
        links = pair_scores[pair_scores["score"] > t]
        if apply_cap:
            preds = apply_reuse_cap(links)
        else:
            preds = {}
            for (id1, id2), _score in links:
                preds.setdefault(id1, set()).add(id2)
        score = macro_f_half(preds, gt_map, all_s1_ids)
        if score > best_score:
            best_score, best_t = score, float(t)
    return best_t, best_score

best_t_base, best_score_base = tune_threshold(pair_scores, gt_map, val_s1_ids, False)
links_base = pair_scores[pair_scores["score"] > best_t_base]
preds_base = {}
for (id1, id2), _score in links_base:
    preds_base.setdefault(id1, set()).add(id2)
print(f"Baseline F_0.5 = {best_score_base:.4f} (threshold={best_t_base:.2f})")

# REUSE-CAP
print("\n--- REUSE-CAP ONLY ---")
best_t_cap, best_score_cap = tune_threshold(pair_scores, gt_map, val_s1_ids, True)
links_cap = pair_scores[pair_scores["score"] > best_t_cap]
preds_cap = apply_reuse_cap(links_cap)
print(f"Reuse-Cap F_0.5 = {best_score_cap:.4f} (threshold={best_t_cap:.2f})")
report_dropped_pairs(preds_base, preds_cap, gt_map, "reuse-cap vs baseline")

# REUSE-CAP + FILTER
print("\n--- REUSE-CAP + MIN-NAME-LEN FILTER ---")
name_by_id = {}
for df_src in (s1_val, s2, s3):
    for r in df_src.itertuples(index=False):
        name_by_id[r.entity_id] = r.name_norm
preds_cap_filt, _, _ = apply_min_name_len_filter(preds_cap, name_by_id, 4)
score_cap_filt = macro_f_half(preds_cap_filt, gt_map, val_s1_ids)
print(f"Reuse-Cap + Filter F_0.5 = {score_cap_filt:.4f} (threshold={best_t_cap:.2f})")
report_dropped_pairs(preds_cap, preds_cap_filt, gt_map, "filter vs cap")
report_dropped_pairs(preds_base, preds_cap_filt, gt_map, "cap+filter vs baseline")




