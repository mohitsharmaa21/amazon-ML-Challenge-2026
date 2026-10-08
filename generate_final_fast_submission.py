import os
import sys
import time
import gc
import re
import math
from collections import defaultdict
import numpy as np

# Set low priority and limit threads to guarantee zero PC lag
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

ABBREV_MAP = {
    'corp': 'corporation', 'co': 'company', 'inc': 'incorporated',
    'ltd': 'limited', 'pvt': 'private', 'llp': 'limited liability partnership',
    'llc': 'limited liability company', 'rd': 'road', 'st': 'street',
    'ave': 'avenue', 'blvd': 'boulevard', 'apt': 'apartment',
    'fl': 'floor', 'no': 'number', 'nr': 'near', 'opp': 'opposite',
    'hno': 'number', 'ste': 'suite', 'dr': 'drive', 'ln': 'lane'
}

RE_NON_ALPHANUM = re.compile(r'[^a-z0-9\s]')
RE_SPACES = re.compile(r'\s+')
RE_URL = re.compile(r'www\.[a-z0-9\-]+\.(com|org|net|in|fr|io)')

def tokenize(name, addr):
    s = f"{name} {addr}".lower()
    s = s.replace('&', ' and ')
    s = RE_URL.sub(' ', s)
    s = RE_NON_ALPHANUM.sub(' ', s)
    tokens = [ABBREV_MAP.get(w, w) for w in RE_SPACES.sub(' ', s).strip().split() if len(w) > 1]
    return tokens

def read_source_by_country(filepath, target_country):
    records = []
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
                records.append((eid, bname, baddr))
    return records

def process_country(country, test_dir):
    print(f"\n=================================================================")
    print(f"--> PROCESSING COUNTRY: {country.upper()}")
    print(f"=================================================================")
    t0 = time.time()
    
    # 1. Load S1 and S23 records
    s1_raw = read_source_by_country(os.path.join(test_dir, "test_source1.tsv"), country)
    print(f"Loaded {len(s1_raw):,} Source 1 records for {country} in {time.time()-t0:.2f}s.")
    
    t_load = time.time()
    s2_raw = read_source_by_country(os.path.join(test_dir, "test_source2.tsv"), country)
    s3_raw = read_source_by_country(os.path.join(test_dir, "test_source3.tsv"), country)
    s23_raw = s2_raw + s3_raw
    del s2_raw, s3_raw
    print(f"Loaded {len(s23_raw):,} S2/S3 records for {country} in {time.time()-t_load:.2f}s.")
    
    # 2. Tokenize and compute IDF
    t_idf = time.time()
    doc_freq = defaultdict(int)
    
    s1_toks_list = []
    for eid, bname, baddr in s1_raw:
        toks = tokenize(bname, baddr)
        s1_toks_list.append((eid, toks))
        for t in set(toks):
            doc_freq[t] += 1
            
    s23_toks_list = []
    for eid, bname, baddr in s23_raw:
        toks = tokenize(bname, baddr)
        s23_toks_list.append((eid, toks))
        for t in set(toks):
            doc_freq[t] += 1
            
    del s1_raw, s23_raw
    gc.collect()
    
    N = len(s1_toks_list) + len(s23_toks_list)
    idf = {t: math.log((N + 1) / (df + 1)) + 1.0 for t, df in doc_freq.items()}
    
    # 3. Vectorize S23 and build Inverted Index
    print("Building Inverted Index for S23...")
    s23_index = defaultdict(list)
    s23_vecs = []
    
    # Cap common tokens to avoid long lists
    MAX_POSTING = 2000
    
    for idx, (eid, toks) in enumerate(s23_toks_list):
        tf_idf = {t: idf[t] for t in toks}
        norm = math.sqrt(sum(v * v for v in tf_idf.values())) if tf_idf else 1.0
        s23_vecs.append((eid, tf_idf, norm))
        for t in set(toks):
            if doc_freq[t] <= MAX_POSTING:
                s23_index[t].append(idx)
                
    del s23_toks_list
    gc.collect()
    print(f"Index ready in {time.time()-t_idf:.2f}s. Indexed {len(s23_vecs):,} records.")
    
    # 4. Fast Match S1 against S23
    t_match = time.time()
    CAND_FLOOR = 0.22
    CAND_TOP_K = 15
    MATCH_THRESH = 0.46
    
    matches_scored = []
    country_candidates = {}
    
    total_s1 = len(s1_toks_list)
    for i, (eid, toks1) in enumerate(s1_toks_list):
        tf_idf1 = {t: idf[t] for t in toks1}
        norm1 = math.sqrt(sum(v * v for v in tf_idf1.values())) if tf_idf1 else 1.0
        
        cand_indices = set()
        for t in set(toks1):
            if t in s23_index:
                cand_indices.update(s23_index[t])
                
        # Score candidates
        scored = []
        for c_idx in cand_indices:
            s2_eid, tf_idf2, norm2 = s23_vecs[c_idx]
            common_toks = set(tf_idf1) & set(tf_idf2)
            if common_toks:
                dot = sum(tf_idf1[t] * tf_idf2[t] for t in common_toks)
                cos_sim = dot / (norm1 * norm2)
                if cos_sim >= CAND_FLOOR:
                    scored.append((cos_sim, s2_eid))
                    
        if scored:
            scored.sort(key=lambda x: -x[0])
            top_cands = scored[:CAND_TOP_K]
            country_candidates[eid] = [x[1] for x in top_cands]
            for sc, s2_eid in top_cands:
                if sc >= MATCH_THRESH:
                    matches_scored.append((sc, eid, s2_eid))
                    
        if (i + 1) % 100000 == 0 or (i + 1) == total_s1:
            print(f"  [{country.upper()}] Processed {i+1:,}/{total_s1:,} S1 records ({time.time()-t_match:.1f}s)...")
            
    print(f"Scoring completed for {country.upper()} in {time.time()-t_match:.2f}s. Scored {len(matches_scored):,} matches.")
    
    # 5. Greedy 1-to-many Reuse Cap
    print(f"Applying greedy reuse cap for {country.upper()}...")
    s2_best = {}
    for sc, s1_id, s2_id in matches_scored:
        if s2_id not in s2_best or sc > s2_best[s2_id][0]:
            s2_best[s2_id] = (sc, s1_id)
            
    country_matches = defaultdict(set)
    for s2_id, (sc, s1_id) in s2_best.items():
        country_matches[s1_id].add(s2_id)
        
    n_m = len(country_matches)
    print(f"Done {country.upper()}: {n_m:,} / {total_s1:,} ({n_m/total_s1*100:.2f}%) entities matched in {time.time()-t0:.2f}s total.")
    
    del s1_toks_list, s23_vecs, s23_index, doc_freq, idf, matches_scored, s2_best
    gc.collect()
    
    return country_matches, country_candidates

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
    
    for country in ["france", "us", "india"]:
        matches, cands = process_country(country, test_dir)
        all_final_matches.update(matches)
        all_candidates.update(cands)
        
    print("\n=======================================================")
    print("WRITING FINAL SUBMISSION FILES TO DISK...")
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
