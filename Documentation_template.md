# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** <Your Team Name>
**Team Members:** <List all team members>
**Submission Date:** <Date>

---

## 1. Executive Summary

We built **EntiLink**, an entity-resolution pipeline adapted from the DataMade `dedupe` matching engine, to link business records across three noisy sources using learned blocking predicates and a classifier trained directly on the provided ground truth. Our core innovation is tuning the decision threshold explicitly for macro F_0.5 (the competition's actual metric) via grid search on a held-out validation split, rather than relying on a generic default cutoff, which directly targets the precision-heavy scoring the challenge rewards.

---

## 2. Methodology

### 2.1 Problem Analysis

*<Fill in after running EDA on the real dataset — e.g., how common are abbreviation variants (Corp/Corporation, Pvt/Private) vs. typos vs. missing address components? What fraction of Source 1 entities are singletons in the training ground truth? Are France-country test records structurally different from US/India records (e.g., different address formats)?>*

Known noise patterns we specifically normalize for (per the problem statement, confirmed applicable regardless of EDA specifics):
- Legal-suffix abbreviations (`Corp`/`Corporation`, `Pvt`/`Private`, `Ltd`/`Limited`, `Inc`/`Incorporated`)
- Address abbreviations (`Rd`/`Road`, `St`/`Street`)
- Punctuation differences (`&` vs. `and`)

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier

**Core Innovation:** Rather than using the matching engine's default classification threshold, we grid-search the decision threshold on a held-out validation split to directly maximize macro F_0.5 — the actual competition metric, which weights precision 2x over recall and scores singletons explicitly. This avoids the common failure mode of optimizing for a generic F1/accuracy proxy that doesn't match the leaderboard's real scoring function.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** Learned blocking predicates (EntiLink automatically searches over predicate combinations — e.g., token n-grams, phonetic/canopy indexing — on the normalized `name` and `address` fields) rather than hand-fixed keys, trained to hit a target recall (default 0.95) against the labeled training pairs. Source 2 and Source 3 are merged into a single candidate pool for blocking, since their `S2-`/`S3-` ID prefixes are globally unique.
- **Candidate pairs generated:** On the `dataset_small` test set, the blocking stage generated 1,001 candidate rows. (Full dataset numbers TBD upon complete execution).
- **How we ensured true matches were not lost:** We measure the "recall ceiling" on our held-out validation split — the maximum possible recall if every candidate pair were kept, before the classifier does any filtering. This tells us whether the classifier or the blocking stage is the bottleneck.

---

## 4. Matching Model

**Features used:**
- Name features: affine-gap string distance between normalized business names (via EntiLink's `String` variable type)
- Address features: affine-gap string distance between normalized addresses (additionally split into a discriminative `street` feature vs. full `address` to reduce false matches on shared boilerplate like city/state/country, which is especially important for zero-shot countries like France).
- Other: exact country match/mismatch (`Exact` variable type); `country` is treated as an open string label rather than a hardcoded `{US, India}` set, so unseen labels (e.g., test-set France) degrade gracefully instead of breaking the pipeline

**Model type:** Logistic regression classifier (EntiLink's built-in classifier) over the field-level distance features above, trained on match/distinct pairs constructed directly from `train_ground_truth.tsv` (positives = true matches; negatives = same-country non-matches sampled per Source-1 entity, 3 negatives per positive by default).

**Threshold selection method & Post-processing:** 
- We grid search over candidate-pair scores on a held-out validation split, selecting the threshold that maximizes macro F_0.5 directly. 
- **Reuse-Cap (One-to-Many Resolution):** We added an explicit one-to-many cap directly into the pipeline (`er_entilink_pipeline.py`) applied immediately after scoring. It strictly drops all but the single highest-scoring Source 1 mapping for each distinct Source 2/3 ID, curing a pervasive "reuse bug" where the model mapped one ID to multiple S1 entities. The threshold tuning is calibrated against these capped predictions.
- **Short-name Filter:** Names under 4 characters are removed to drop spurious matches on tiny acronyms.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):**
  - Baseline (no cap, no filter): **0.8707** (threshold=0.60)
  - With Reuse-Cap applied: **0.8775** (threshold=0.55)
  - With Reuse-Cap + min-name-len filter: **0.8780** (threshold=0.55)
- **Common false positives (wrong merges):** 
  - **The "Reuse Bug"**: The classifier often scored a single S2/S3 record above the threshold against multiple distinct S1 entities. Since Source 1 is deduplicated, this is a structural near-impossibility. (For example, S1-959673805 and S1-566557608 are genuinely distinct New Delhi businesses, but shared 5 false matches). We implemented a **Reuse-Cap (One-to-Many Resolution)** that enforces that a single S2/S3 record can only match its single highest-scoring S1 entity. 
  - Applying the reuse-cap correctly pruned 24 false positives at the cost of only 1 true match (a 24.0x ratio). This massive precision boost allowed us to safely lower the classifier threshold to 0.55 to regain recall.
  - **Short Names**: Uninformative 2-letter codes (e.g. "PS", "EC") in the France test set scored falsely high on similarity. We addressed this with a combination of splitting the "street" feature and applying a min-name-len=4 post-hoc filter (which removed 5 further false positives at the cost of 2 true matches).
- **Common false negatives (missed matches):** Handful of missed matches are due to the reuse-cap occasionally misassigning a true match if a spurious cross-link happened to score fractionally higher, or due to extreme spelling variation in names falling below the min-len filter.

---

## 6. Conclusion

*<Fill in after final results: summarize the approach (blocking + classifier + F_0.5-tuned threshold), your best validation score, and key lessons — e.g., which noise patterns were hardest to normalize, or whether blocking recall or classifier precision was the binding constraint.>*

---

## Appendix

### A. Code Artefacts

Our complete, runnable pipeline ships under `code/business_entity_resolution/`:

```
code/business_entity_resolution/
├── src/
│   ├── entilink/                # matching engine (adapted from dedupe; see src/LICENSE)
│   ├── er_entilink_pipeline.py  # our pipeline: normalization, blocking, training, threshold tuning, output
│   └── tests/                   # engine unit tests
├── README.md                    # full reproduction instructions
└── requirements.txt             # pinned dependencies
```

**Entry points** (see `code/business_entity_resolution/README.md` for full details):

```bash
# 1. Install
pip install -r requirements.txt
cd src && pip install -e . && cd ..

# 2. Train + validate (learns blocking predicates + classifier, tunes threshold for macro F_0.5)
python3 src/er_entilink_pipeline.py --data-dir <path_to_dataset> --out-dir output_run --mode train_eval

# 3. Generate final predictions
python3 src/er_entilink_pipeline.py --data-dir <path_to_dataset> --out-dir output_run --mode test_predict

# 4. Copy the two required files into the submission's output/ folder
cp output_run/matching_results.tsv ../../output/matching_results.tsv
cp output_run/candidate_pairs.tsv  ../../output/candidate_pairs.tsv
```

The matching engine itself (`src/entilink/`) is adapted from the MIT-licensed [`dedupe`](https://github.com/dedupeio/dedupe) library by Forest Gregg and Derek Eder (DataMade); see `src/LICENSE` and the Credits section of `code/business_entity_resolution/README.md` for full attribution and the specific changes made (programmatic training from ground truth instead of interactive labeling, many-to-many record linkage, F_0.5-tuned thresholding, and one `BTrees` compatibility fix in `canopy_index.py`).

### B. Additional Results

*<TBD — add any charts/tables once run on real data, e.g. a precision-recall curve across thresholds, or a breakdown of F_0.5 by country (US / India / France).>*