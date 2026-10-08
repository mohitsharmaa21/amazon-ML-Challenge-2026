# Business Entity Resolution — EntiLink

Reproduction instructions: data → blocking → matching → output.

## What's here

```
business_entity_resolution/
├── src/
│   ├── entilink/                # matching engine (adapted from dedupe; see src/LICENSE + Credits below)
│   ├── er_entilink_pipeline.py  # our task-specific pipeline (blocking, training, threshold tuning, output)
│   ├── tests/                   # engine unit tests (inherited from upstream, unmodified logic)
│   ├── pyproject.toml / setup.py / MANIFEST.in
│   └── LICENSE
├── README.md                    # this file
└── requirements.txt             # pinned dependencies
```

## 1. Environment setup

```bash
cd src/
pip install -r ../requirements.txt
pip install -e .          # builds the Cython extension (entilink/cpredicates.pyx) and installs `entilink`
python3 -c "import entilink; print(entilink.__file__)"   # sanity check
```

## 2. Expected data layout

Place the competition data as:

```
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

## 3. Reproduce end-to-end

**Step 1 — Train + validate** (splits training Source-1 entities 80/20, builds candidate pairs via EntiLink's learned blocking predicates, trains the pairwise classifier from `train_ground_truth.tsv`, tunes the decision threshold for macro F_0.5 on the held-out split, and saves the trained model):

```bash
python3 src/er_entilink_pipeline.py \
    --data-dir dataset \
    --out-dir output \
    --mode train_eval
```

This prints the validation macro F_0.5 and writes `output/val_matching_results.tsv`, `output/entilink_settings.bin`, and `output/threshold.txt`.

**Step 2 — Generate final test predictions** (loads the saved model/threshold, runs blocking + scoring on the test set, and writes both required submission files):

```bash
python3 src/er_entilink_pipeline.py \
    --data-dir dataset \
    --out-dir output \
    --mode test_predict
```

This produces:
- `output/candidate_pairs.tsv` — every Source-2/3 record EntiLink's blocking stage considered a plausible candidate for each Source-1 entity (the exact set fed to the classifier)
- `output/matching_results.tsv` — final matches after thresholding, in the required `source1_entity_id \t matched_entity_ids` format

## 4. Pipeline summary (data → blocking → matching → output)

1. **Normalize** — lowercase, strip punctuation, expand legal-suffix and address abbreviations (`Corp`→`corporation`, `Rd`→`road`, `Pvt Ltd`→`private limited`, etc.) for `business_name` and `business_address`.
2. **Block** — EntiLink (`RecordLink`) learns blocking predicates from labeled training pairs, indexing Source 2 + Source 3 together (their `S2-`/`S3-` IDs are globally unique) and generating candidate pairs per Source-1 entity.
3. **Match** — a classifier trained on the ground-truth match/distinct pairs scores each candidate pair; `.join(..., constraint="many-to-many")` is used since one Source-1 entity may match several Source-2/3 records.
4. **Threshold** — the score cutoff is grid-searched on a held-out validation split to maximize **macro F_0.5**, the competition's actual metric (not dedupe's default 0.5 cutoff).
5. **Output** — predictions above threshold are written to `matching_results.tsv`; every candidate considered (pre-threshold) is written to `candidate_pairs.tsv`.

## Credits

The underlying matching/blocking engine (`src/entilink/`) is adapted from **[`dedupe`](https://github.com/dedupeio/dedupe)** by Forest Gregg and Derek Eder (DataMade), MIT licensed. See `src/LICENSE` and `src/entilink/` docstrings for full attribution. One compatibility fix was applied in `src/entilink/canopy_index.py` (a newer `BTrees` release removed the `.byValue()` method the original canopy-index code relied on); everything else in `src/entilink/` is the unmodified upstream matching engine. The task-specific adaptation layer (`src/er_entilink_pipeline.py`) is original work for this challenge.
