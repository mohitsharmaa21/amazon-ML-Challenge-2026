import os
import sys
import time
import gc
import re
import random
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
random.seed(42)
np.random.seed(42)

ABBREV_MAP = {
    r"\bcorp\b": "corporation", r"\bco\b": "company", r"\binc\b": "incorporated",
    r"\bltd\b": "limited",      r"\bpvt\b": "private",
    r"\bllp\b": "limited liability partnership",
    r"\bllc\b": "limited liability company",
    r"\brd\b": "road",  r"\bst\b": "street", r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard", r"\bapt\b": "apartment",
    r"\bfl\b": "floor", r"\bno\b": "number", r"\bnr\b": "near", r"\bopp\b": "opposite",
    r"\bhno\b": "number", r"\bste\b": "suite", r"\bdr\b": "drive", r"\bln\b": "lane"
}

RE_URL = re.compile(r"www\.[a-z0-9\-]+\.(com|org|net|in|fr|io)")
RE_NON_ALPHANUM = re.compile(r"[^\w\s]")
RE_SPACES = re.compile(r"\s+")

def clean_text(name, addr):
    s = f"{name} {addr}".lower()
    s = s.replace("&", " and ")
    s = RE_URL.sub(" ", s)
    s = RE_NON_ALPHANUM.sub(" ", s)
    s = RE_SPACES.sub(" ", s).strip()
    for pat, repl in ABBREV_MAP.items():
        s = re.sub(pat, repl, s)
    return re.sub(r"\s+", " ", s).strip()

def stream_source_chunks(filepath, target_country, chunk_size=100000):
    """Yield chunks of (ids, texts) for the given country without loading entire file."""
    ids, texts = [], []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4:
                eid, bname, baddr, country = parts[0], parts[1], parts[2], parts[3]
            elif len(parts) == 3:
                eid, bname, baddr, country = parts[0], parts[1], parts[2], ""
            else:
                continue
            if country.lower().strip() == target_country:
                ids.append(eid)
                texts.append(clean_text(bname, baddr))
                if len(ids) >= chunk_size:
                    yield ids, texts
                    ids, texts = [], []
    if ids:
        yield ids, texts

def read_all_country_s1(filepath, target_country):
    ids, texts = [], []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4:
                eid, bname, baddr, country = parts[0], parts[1], parts[2], parts[3]
            elif len(parts) == 3:
                eid, bname, baddr, country = parts[0], parts[1], parts[2], ""
            else:
                continue
            if country.lower().strip() == target_country:
                ids.append(eid)
                texts.append(clean_text(bname, baddr))
    return ids, texts

def main():
    t_start = time.time()
    base_dir = r"c:\Users\kisha\Downloads\team_name_submission\team_name_submission"
    test_dir = os.path.join(base_dir, "code", "business_entity_resolution", "dataset", "test")
    
    out_dirs = [
        os.path.join(base_dir, "output"),
        os.path.join(base_dir, "output_run"),
        os.path.join(base_dir, "code", "business_entity_resolution", "output"),
        base_dir
    ]
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
        
    print("Reading required Test Source 1 IDs in exact order...")
    all_s1_ids = []
    with open(os.path.join(test_dir, "test_source1.tsv"), "r", encoding="utf-8", errors="replace") as f:
        f.readline()
        for line in f:
            parts = line.split("\t", 1)
            if parts and parts[0].strip():
                all_s1_ids.append(parts[0].strip())
    print(f"Total required Test Source 1 entities: {len(all_s1_ids):,}")
    
    all_final_matches = defaultdict(set)
    all_candidates = defaultdict(list)
    
    # Memory-Safe Knobs (Peak RAM < 1.8 GB)
    CAND_FLOOR = 0.22
    CAND_TOP_K = 15
    MATCH_THRESH = 0.48
    MAX_FEATURES = 50000
    S23_CHUNK_SIZE = 100000
    S1_BLOCK_SIZE = 25000
    
    for country in ["france", "us", "india"]:
        print(f"\n{'='*65}")
        print(f"--> PROCESSING COUNTRY: {country.upper()}")
        print(f"{'='*65}")
        t0 = time.time()
        
        s1_ids, s1_texts = read_all_country_s1(os.path.join(test_dir, "test_source1.tsv"), country)
        n_s1 = len(s1_ids)
        print(f"Loaded {n_s1:,} Source 1 records for {country} in {time.time()-t0:.2f}s.")
        
        # Sample for vectorizer
        print("Fitting compact TF-IDF vectorizer...")
        sample_pool = random.sample(s1_texts, min(50000, len(s1_texts)))
        for s_file in ["test_source2.tsv", "test_source3.tsv"]:
            for _, c_texts in stream_source_chunks(os.path.join(test_dir, s_file), country, chunk_size=25000):
                sample_pool += c_texts
                if len(sample_pool) >= 100000:
                    break
            if len(sample_pool) >= 100000:
                break
                
        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=MAX_FEATURES, sublinear_tf=True, dtype=np.float32)
        vec.fit(sample_pool)
        del sample_pool; gc.collect()
        
        # Vectorize S1 in blocks to keep RAM tiny
        print(f"Vectorizing S1 records ({n_s1:,} items)...")
        X_s1 = vec.transform(s1_texts)
        del s1_texts; gc.collect()
        
        # Store Top-K candidates per S1 entity: list of (score, s2_id)
        # Using flat dict of lists: memory for 800k entities * 15 tuples is ~100MB
        top_candidates = [[] for _ in range(n_s1)]
        s2_best_match = {} # s2_id -> (score, s1_id)
        
        # Stream S2 and S3 in chunks of 100k
        t_match = time.time()
        total_s23_processed = 0
        
        for s_filename in ["test_source2.tsv", "test_source3.tsv"]:
            s_path = os.path.join(test_dir, s_filename)
            print(f"Streaming {s_filename} for {country.upper()}...")
            
            for chunk_ids, chunk_texts in stream_source_chunks(s_path, country, chunk_size=S23_CHUNK_SIZE):
                chunk_n = len(chunk_ids)
                total_s23_processed += chunk_n
                
                X_chunk = vec.transform(chunk_texts)
                del chunk_texts; gc.collect()
                
                # Match in S1 blocks against this S23 chunk
                for b_start in range(0, n_s1, S1_BLOCK_SIZE):
                    b_end = min(b_start + S1_BLOCK_SIZE, n_s1)
                    sim_sparse = (X_s1[b_start:b_end] @ X_chunk.T).tocsr()
                    
                    # In-place filter below floor
                    sim_sparse.data[sim_sparse.data < CAND_FLOOR] = 0
                    sim_sparse.eliminate_zeros()
                    
                    indptr = sim_sparse.indptr
                    indices = sim_sparse.indices
                    data = sim_sparse.data
                    
                    for r_idx in range(b_end - b_start):
                        global_r = b_start + r_idx
                        start = indptr[r_idx]
                        end = indptr[r_idx + 1]
                        if start < end:
                            for pos in range(start, end):
                                sc = float(data[pos])
                                s2_id = chunk_ids[indices[pos]]
                                
                                # Add to candidate list
                                top_candidates[global_r].append((sc, s2_id))
                                
                                # Track best match for reuse cap
                                if sc >= MATCH_THRESH:
                                    s1_id = s1_ids[global_r]
                                    if s2_id not in s2_best_match or sc > s2_best_match[s2_id][0]:
                                        s2_best_match[s2_id] = (sc, s1_id)
                                        
                    del sim_sparse
                    
                # Periodic prune to keep candidate list small
                if total_s23_processed % 500000 == 0:
                    for i in range(n_s1):
                        if len(top_candidates[i]) > CAND_TOP_K * 2:
                            top_candidates[i].sort(key=lambda x: -x[0])
                            top_candidates[i] = top_candidates[i][:CAND_TOP_K]
                            
                print(f"  Processed {total_s23_processed:,} S2/S3 records ({time.time()-t_match:.1f}s)...")
                del X_chunk, chunk_ids; gc.collect()
                
        # Finalize candidates and matches for this country
        print("Finalizing country matches and candidate sets...")
        for i in range(n_s1):
            eid = s1_ids[i]
            cands = top_candidates[i]
            if cands:
                cands.sort(key=lambda x: -x[0])
                all_candidates[eid] = [c[1] for c in cands[:CAND_TOP_K]]
                
        for s2_id, (sc, s1_id) in s2_best_match.items():
            all_final_matches[s1_id].add(s2_id)
            
        n_m = sum(1 for sid in s1_ids if sid in all_final_matches)
        print(f"Done {country.upper()}: {n_m:,} / {n_s1:,} ({n_m/n_s1*100:.2f}%) entities matched in {time.time()-t0:.2f}s total.")
        
        del s1_ids, X_s1, top_candidates, s2_best_match, vec
        gc.collect()
        
    print("\n=======================================================")
    print("WRITING SUBMISSION FILES TO DISK...")
    print("=======================================================")
    for out_dir in out_dirs:
        m_file = os.path.join(out_dir, "matching_results.tsv")
        c_file = os.path.join(out_dir, "candidate_pairs.tsv")
        print(f"Writing {m_file}...")
        with open(m_file, "w", encoding="utf-8", newline="") as fm:
            fm.write("source1_entity_id\tmatched_entity_ids\n")
            for sid in all_s1_ids:
                m_set = all_final_matches.get(sid, set())
                fm.write(f"{sid}\t{','.join(sorted(m_set))}\n")
                
        print(f"Writing {c_file}...")
        with open(c_file, "w", encoding="utf-8", newline="") as fc:
            fc.write("source1_entity_id\tcandidate_entity_ids\n")
            for sid in all_s1_ids:
                c_list = all_candidates.get(sid, [])
                m_set = all_final_matches.get(sid, set())
                combined_cands = list(dict.fromkeys(c_list + list(m_set)))
                fc.write(f"{sid}\t{','.join(combined_cands)}\n")
                
    print(f"\n=======================================================")
    print(f"ALL JOBS FINISHED SUCCESSFULLY in {time.time()-t_start:.2f}s ({((time.time()-t_start)/60):.1f} mins)!")
    print(f"=======================================================")

if __name__ == "__main__":
    main()
