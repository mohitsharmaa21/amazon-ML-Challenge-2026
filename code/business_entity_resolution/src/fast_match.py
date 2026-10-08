"""
fast_match.py  -- Memory-safe TF-IDF entity resolution
Designed for 16GB RAM laptop. Fits vectorizer on a SAMPLE to avoid OOM.
Uses word-level (1-2) grams, max_features=40000, BATCH_S1=1000, BATCH_S23=30000.
Peak memory ~5GB. Runs in ~60-90 minutes.
"""
import gc, re, random
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# --- tuneable knobs (sized for 16GB RAM, 8GB available) ---
SAMPLE_FIT   = 400_000   # names sampled to FIT the vectorizer (avoids OOM)
MAX_FEATURES = 40_000    # vocabulary cap (keeps sparse matrices small)
NAME_THRESH  = 0.25      # minimum name cosine to keep as candidate
FINAL_THRESH = 0.50      # minimum combined score to call a match
TOP_K        = 8         # max candidates stored per S1 entity
BATCH_S1     = 1_000     # S1 rows per outer batch
BATCH_S23    = 30_000    # S23 rows per inner batch  => 1000*30k*4B = 120MB per dense block
MIN_NAME_LEN = 4
RANDOM_SEED  = 42
# ----------------------------------------------------------

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

ROOT     = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "code" / "business_entity_resolution" / "dataset"
OUT_DIR  = ROOT / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ABBREV_MAP = {
    r"\bcorp\b":"corporation", r"\bco\b":"company", r"\binc\b":"incorporated",
    r"\bltd\b":"limited",      r"\bpvt\b":"private",
    r"\bllp\b":"limited liability partnership",
    r"\bllc\b":"limited liability company",
    r"\brd\b":"road",  r"\bst\b":"street", r"\bave\b":"avenue",
    r"\bblvd\b":"boulevard", r"\bapt\b":"apartment",
    r"\bfl\b":"floor", r"\bno\b":"number", r"\bnr\b":"near", r"\bopp\b":"opposite",
}

def normalize(s):
    if not isinstance(s, str): s = "" if pd.isna(s) else str(s)
    s = s.lower().strip().replace("&", " and ")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+",     " ", s).strip()
    for pat, repl in ABBREV_MAP.items():
        s = re.sub(pat, repl, s)
    return re.sub(r"\s+", " ", s).strip()

def load_source(path):
    print(f"  loading {path.name} ...", flush=True)
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    df["name_norm"]   = df["business_name"].apply(normalize)
    df["street_norm"] = df["business_address"].str.split(",").str[0].apply(normalize)
    df["country_lc"]  = df["country"].str.lower().str.strip()
    return df

# ── 1. Load all data ─────────────────────────────────────────────────────────
print("Loading sources ...", flush=True)
s1  = load_source(DATA_DIR / "test" / "test_source1.tsv")
s23 = pd.concat([
    load_source(DATA_DIR / "test" / "test_source2.tsv"),
    load_source(DATA_DIR / "test" / "test_source3.tsv"),
], ignore_index=True)
print(f"S1={len(s1):,}  S23={len(s23):,}", flush=True)

# ── 2. Fit vectorizers on a SAMPLE (avoids OOM during fit) ───────────────────
print(f"Sampling {SAMPLE_FIT:,} names for TF-IDF fit ...", flush=True)
all_names   = pd.concat([s1["name_norm"],   s23["name_norm"]],   ignore_index=True)
all_streets = pd.concat([s1["street_norm"], s23["street_norm"]], ignore_index=True)

idx = random.sample(range(len(all_names)), min(SAMPLE_FIT, len(all_names)))
sample_names   = all_names.iloc[idx].tolist()
sample_streets = all_streets.iloc[idx].tolist()
del all_names, all_streets; gc.collect()

print("Fitting name vectoriser ...", flush=True)
vec_name = TfidfVectorizer(
    analyzer="word", ngram_range=(1, 2),
    max_features=MAX_FEATURES, min_df=1, sublinear_tf=True
)
vec_name.fit(sample_names); del sample_names; gc.collect()

print("Fitting address vectoriser ...", flush=True)
vec_addr = TfidfVectorizer(
    analyzer="word", ngram_range=(1, 2),
    max_features=MAX_FEATURES, min_df=1, sublinear_tf=True
)
vec_addr.fit(sample_streets); del sample_streets; gc.collect()
print("Vectorisers ready.", flush=True)

# ── 3. Match per country ──────────────────────────────────────────────────────
# Collect ALL pairs above NAME_THRESH, apply reuse-cap + final threshold at end
all_candidates = {}   # s1_id -> set of s23_ids (for candidate_pairs.tsv)
all_scored     = []   # (score, s1_id, s23_id) for matched pairs

countries = sorted(set(s1["country_lc"].unique()) | set(s23["country_lc"].unique()))
print(f"Countries: {countries}", flush=True)

for country in countries:
    s1_c  = s1 [s1 ["country_lc"] == country].reset_index(drop=True)
    s23_c = s23[s23["country_lc"] == country].reset_index(drop=True)
    if len(s1_c) == 0 or len(s23_c) == 0: continue
    print(f"\n[{country}]  S1={len(s1_c):,}  S23={len(s23_c):,}", flush=True)

    s23_ids   = s23_c["entity_id"].values
    s23_names = s23_c["name_norm"].values

    # Vectorise entire S23 haystack for this country (sparse — safe)
    print(f"  Vectorising S23 haystack ...", flush=True)
    hay_name = vec_name.transform(s23_c["name_norm"])
    hay_addr = vec_addr.transform(s23_c["street_norm"])

    total_s1 = len(s1_c)
    for s1_start in range(0, total_s1, BATCH_S1):
        s1_end   = min(s1_start + BATCH_S1, total_s1)
        s1_batch = s1_c.iloc[s1_start:s1_end]
        n_s1b    = len(s1_batch)

        q_name = vec_name.transform(s1_batch["name_norm"])
        q_addr = vec_addr.transform(s1_batch["street_norm"])

        # Track best TOP_K per S1 row across all S23 chunks
        best_scores = np.full((n_s1b, TOP_K), -1.0, dtype=np.float32)
        best_gidx   = np.full((n_s1b, TOP_K), -1,   dtype=np.int32)

        n_s23 = len(s23_c)
        for s23_start in range(0, n_s23, BATCH_S23):
            s23_end = min(s23_start + BATCH_S23, n_s23)
            chunk_n = s23_end - s23_start

            # Dense blocks: (n_s1b, chunk_n) — max 120 MB
            sc_name = (q_name @ hay_name[s23_start:s23_end].T).toarray().astype(np.float32)
            sc_addr = (q_addr @ hay_addr[s23_start:s23_end].T).toarray().astype(np.float32)
            combined = (0.70 * sc_name + 0.30 * sc_addr)

            # zero out below name threshold
            combined[sc_name <= NAME_THRESH] = -1.0

            # vectorised update of running top-K
            # stack current bests with this chunk's scores
            all_s = np.concatenate([best_scores, combined], axis=1)  # (n_s1b, TOP_K+chunk_n)
            local_idx  = np.argsort(all_s, axis=1)[:, -(TOP_K):]     # top-K indices in concatenated
            best_scores = np.take_along_axis(all_s, local_idx, axis=1)

            # map local indices back to global S23 indices
            old_k      = TOP_K
            new_gidx   = np.where(local_idx >= old_k,
                                  (local_idx - old_k + s23_start).astype(np.int32),
                                  best_gidx[np.arange(n_s1b)[:, None],
                                            np.minimum(local_idx, old_k - 1)])
            best_gidx  = new_gidx
            del sc_name, sc_addr, combined, all_s, local_idx

        # Commit results for this S1 batch
        s1_batch_ids   = s1_batch["entity_id"].values
        s1_batch_names = s1_batch["name_norm"].values
        for i in range(n_s1b):
            s1_id   = s1_batch_ids[i]
            s1_name = s1_batch_names[i]
            if len(s1_name) < MIN_NAME_LEN: continue
            for k in range(TOP_K):
                gi = best_gidx[i, k]
                sc = best_scores[i, k]
                if gi < 0 or sc < NAME_THRESH: continue
                s23id = s23_ids[gi]
                all_candidates.setdefault(s1_id, set()).add(s23id)
                if sc >= FINAL_THRESH and len(s23_names[gi]) >= MIN_NAME_LEN:
                    all_scored.append((float(sc), s1_id, s23id))

        if (s1_start // BATCH_S1) % 10 == 0:
            print(f"  {s1_end:,}/{total_s1:,} S1 rows done", flush=True)

    del hay_name, hay_addr, s1_c, s23_c; gc.collect()

# ── 4. Apply reuse-cap: each S23 id -> only best single S1 ───────────────────
print("\nApplying reuse cap ...", flush=True)
best_for_s23 = {}
for sc, s1_id, s23_id in all_scored:
    prev = best_for_s23.get(s23_id)
    if prev is None or sc > prev[0]:
        best_for_s23[s23_id] = (sc, s1_id)

match_dict = {}
for s23_id, (sc, s1_id) in best_for_s23.items():
    match_dict.setdefault(s1_id, set()).add(s23_id)

# ── 5. Write output files ─────────────────────────────────────────────────────
print("Writing output files ...", flush=True)
all_s1_ids = s1["entity_id"].tolist()

cand_rows  = [(sid, ",".join(sorted(all_candidates.get(sid, set())))) for sid in all_s1_ids]
match_rows = [(sid, ",".join(sorted(match_dict.get(sid, set()))))     for sid in all_s1_ids]

pd.DataFrame(cand_rows,  columns=["source1_entity_id", "candidate_entity_ids"]
    ).to_csv(OUT_DIR / "candidate_pairs.tsv",  sep="\t", index=False)
pd.DataFrame(match_rows, columns=["source1_entity_id", "matched_entity_ids"]
    ).to_csv(OUT_DIR / "matching_results.tsv", sep="\t", index=False)

n_matched = sum(1 for _, m in match_rows if m)
n_cands   = sum(1 for _, c in cand_rows  if c)
print(f"\nDone!")
print(f"  {n_matched:,} / {len(all_s1_ids):,} S1 entities matched")
print(f"  {n_cands:,}   / {len(all_s1_ids):,} S1 entities have candidates")
print(f"  Files written to: {OUT_DIR}")