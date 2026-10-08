"""
superfast_match.py
Uses pure sparse matrix dot-products and stops materializing dense matrices.
Streams output directly to files to prevent data loss.
"""
import gc, re, time, random
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

NAME_THRESH  = 0.25
FINAL_THRESH = 0.50
MIN_NAME_LEN = 4

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

print("Loading sources ...", flush=True)
s1  = load_source(DATA_DIR / "test" / "test_source1.tsv")
s23 = pd.concat([
    load_source(DATA_DIR / "test" / "test_source2.tsv"),
    load_source(DATA_DIR / "test" / "test_source3.tsv"),
], ignore_index=True)

print("Fitting TF-IDF on a 400k sample ...", flush=True)
sample_idx = np.random.choice(len(s23), min(400_000, len(s23)), replace=False)
sample_names = s23["name_norm"].iloc[sample_idx]
sample_streets = s23["street_norm"].iloc[sample_idx]

vec_name = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.05, sublinear_tf=True)
vec_name.fit(sample_names)
del sample_names; gc.collect()

vec_addr = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.05, sublinear_tf=True)
vec_addr.fit(sample_streets)
del sample_streets; gc.collect()

countries = sorted(set(s1["country_lc"].unique()) | set(s23["country_lc"].unique()))

f_cand = open(OUT_DIR / "superfast_candidates.tsv", "w", encoding="utf-8")
f_match = open(OUT_DIR / "superfast_matches.tsv", "w", encoding="utf-8")
f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
f_match.write("source1_entity_id\tmatched_entity_ids\n")

for country in countries:
    start_t = time.time()
    s1_c  = s1 [s1 ["country_lc"] == country].reset_index(drop=True)
    s23_c = s23[s23["country_lc"] == country].reset_index(drop=True)
    if len(s1_c) == 0 or len(s23_c) == 0: continue
    
    print(f"\n[{country}]  S1={len(s1_c):,}  S23={len(s23_c):,}", flush=True)
    
    hay_name = vec_name.transform(s23_c["name_norm"])
    hay_addr = vec_addr.transform(s23_c["street_norm"])
    
    s23_ids = s23_c["entity_id"].values
    s23_names = s23_c["name_norm"].values
    
    s1_ids = s1_c["entity_id"].values
    s1_names = s1_c["name_norm"].values
    
    BATCH_S1 = 5000
    total_s1 = len(s1_c)
    
    for s1_start in range(0, total_s1, BATCH_S1):
        s1_end = min(s1_start + BATCH_S1, total_s1)
        batch = s1_c.iloc[s1_start:s1_end]
        
        q_name = vec_name.transform(batch["name_norm"])
        sc_name_sparse = q_name.dot(hay_name.T)
        
        mask = sc_name_sparse.data >= NAME_THRESH
        sc_name_sparse.data[~mask] = 0
        sc_name_sparse.eliminate_zeros()
        
        rows, cols = sc_name_sparse.nonzero()
        
        if len(rows) > 0:
            q_addr = vec_addr.transform(batch["street_norm"])
            q_addr_surv = q_addr[rows]
            hay_addr_surv = hay_addr[cols]
            
            addr_sims = q_addr_surv.multiply(hay_addr_surv).sum(axis=1).A1
            name_sims = sc_name_sparse.data
            
            combined = 0.70 * name_sims + 0.30 * addr_sims
            
            df_pairs = pd.DataFrame({
                "row_idx": rows,
                "s23_idx": cols,
                "score": combined
            })
            
            for r_idx, group in df_pairs.groupby("row_idx"):
                s1_idx = s1_start + r_idx
                s1_id = s1_ids[s1_idx]
                
                if len(s1_names[s1_idx]) < MIN_NAME_LEN:
                    f_cand.write(f"{s1_id}\t\n")
                    f_match.write(f"{s1_id}\t\n")
                    continue
                
                cand_ids = s23_ids[group["s23_idx"].values]
                f_cand.write(f"{s1_id}\t{','.join(sorted(set(cand_ids)))}\n")
                
                valid = group[group["score"] >= FINAL_THRESH]
                valid = valid[valid["s23_idx"].apply(lambda i: len(s23_names[i]) >= MIN_NAME_LEN)]
                
                if len(valid) == 0:
                    f_match.write(f"{s1_id}\t\n")
                    continue
                
                best_idx = valid.loc[valid["score"].idxmax(), "s23_idx"]
                f_match.write(f"{s1_id}\t{s23_ids[best_idx]}\n")
        else:
            for s1_idx in range(s1_start, s1_end):
                f_cand.write(f"{s1_ids[s1_idx]}\t\n")
                f_match.write(f"{s1_ids[s1_idx]}\t\n")

        if (s1_start // BATCH_S1) % 5 == 0:
            print(f"  {s1_end:,}/{total_s1:,} S1 rows done (time: {time.time()-start_t:.1f}s)", flush=True)

f_cand.close()
f_match.close()
print("Done! Files saved as superfast_matches.tsv and superfast_candidates.tsv", flush=True)