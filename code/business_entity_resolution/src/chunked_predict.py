import pandas as pd
import numpy as np
import entilink
from pathlib import Path
import csv
import os
import gc
import re
import argparse

ABBREV_MAP = {
    r"\bcorp\b": "corporation",
    r"\bco\b": "company",
    r"\binc\b": "incorporated",
    r"\bltd\b": "limited",
    r"\bpvt\b": "private",
    r"\bllp\b": "limited liability partnership",
    r"\bllc\b": "limited liability company",
    r"\brd\b": "road",
    r"\bst\b": "street",
    r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bapt\b": "apartment",
    r"\bfl\b": "floor",
    r"\bno\b": "number",
    r"\bnr\b": "near",
    r"\bopp\b": "opposite",
}

def normalize_text(s):
    if pd.isna(s): return ""
    s = str(s).lower().strip()
    s = s.replace("&", " and ")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for pattern, repl in ABBREV_MAP.items():
        s = re.sub(pattern, repl, s)
    return re.sub(r"\s+", " ", s).strip()

def parse_chunk(chunk_df):
    chunk_df = chunk_df.fillna("")
    chunk_df["name_norm"] = chunk_df["business_name"].apply(normalize_text)
    chunk_df["street_raw"] = chunk_df["business_address"].str.split(",").str[0]
    chunk_df["street_norm"] = chunk_df["street_raw"].apply(normalize_text)
    chunk_df["addr_norm"] = chunk_df["business_address"].apply(normalize_text)
    chunk_df["country_norm"] = chunk_df["country"].str.lower().str.strip()
    out = {}
    for row in chunk_df.itertuples(index=False):
        out[row.entity_id] = {
            "name": row.name_norm,
            "street": row.street_norm,
            "address": row.addr_norm,
            "country": row.country_norm,
        }
    return out

def apply_reuse_cap_chunk(links_with_scores):
    best_for_s2 = {}
    for (s1, s2), score in links_with_scores:
        if s2 not in best_for_s2 or score > best_for_s2[s2][0]:
            best_for_s2[s2] = (score, s1)
    
    surviving_links = []
    for s2, (score, s1) in best_for_s2.items():
        surviving_links.append((s1, s2))
    return surviving_links

def main():
    data_dir = Path("../dataset")
    out_dir = Path("../../../output")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading entilink settings...")
    with open(out_dir / "entilink_settings.bin", "rb") as f:
        linker = entilink.StaticRecordLink(f, num_cores=8)
    with open(out_dir / "threshold.txt") as f:
        threshold = float(f.read().strip())
        
    print("Loading test_source1.tsv completely into memory (~700MB)...")
    s1_df = pd.read_csv(data_dir / "test" / "test_source1.tsv", sep="\t", dtype=str)
    s1_all_ids = s1_df["entity_id"].tolist()
    data_1 = parse_chunk(s1_df)
    
    s1_names = {k: v["name"] for k, v in data_1.items()}
    del s1_df
    gc.collect()

    temp_cand_file = out_dir / "temp_cand.csv"
    temp_match_file = out_dir / "temp_match.csv"
    
    with open(temp_cand_file, "w", newline="") as f: pass
    with open(temp_match_file, "w", newline="") as f: pass
    
    def process_s23_file(filepath):
        chunk_size = 500000
        for chunk_idx, chunk_df in enumerate(pd.read_csv(filepath, sep="\t", dtype=str, chunksize=chunk_size)):
            print(f"Processing chunk {chunk_idx+1} of {filepath.name}...")
            chunk_data_2 = parse_chunk(chunk_df)
            
            s2_names = {k: v["name"] for k, v in chunk_data_2.items()}
            
            print("  Generating candidate pairs...")
            pairs = list(linker.pairs(data_1, chunk_data_2))
            
            with open(temp_cand_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for (id1, _), (id2, _) in pairs:
                    writer.writerow([id1, id2])
                    
            print("  Scoring pairs...")
            pair_scores = linker.score(iter(pairs))
            links = pair_scores[pair_scores["score"] > threshold]
            
            raw_links = []
            for (id1, id2), score in links:
                raw_links.append(((id1, id2), score))
                
            print("  Applying reuse cap & filters...")
            capped_links = apply_reuse_cap_chunk(raw_links)
            
            min_len = 4
            final_links = []
            for id1, id2 in capped_links:
                name1 = s1_names.get(id1, "")
                name2 = s2_names.get(id2, "")
                if len(name1) >= min_len and len(name2) >= min_len:
                    final_links.append((id1, id2))
                    
            with open(temp_match_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for id1, id2 in final_links:
                    writer.writerow([id1, id2])
                    
            del chunk_data_2, s2_names, pairs, pair_scores, links, raw_links, capped_links, final_links
            gc.collect()

    process_s23_file(data_dir / "test" / "test_source2.tsv")
    process_s23_file(data_dir / "test" / "test_source3.tsv")
    
    print("Aggregating candidates and matches...")
    cand_dict = {}
    with open(temp_cand_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            cand_dict.setdefault(row[0], set()).add(row[1])
            
    match_dict = {}
    with open(temp_match_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            match_dict.setdefault(row[0], set()).add(row[1])
            
    print("Writing final TSV files...")
    def write_output(preds, out_path, id_col2):
        rows = []
        for s1_id in s1_all_ids:
            ids = sorted(preds.get(s1_id, set()))
            rows.append((s1_id, ",".join(ids)))
        pd.DataFrame(rows, columns=["source1_entity_id", id_col2]).to_csv(out_path, sep="\t", index=False)
        
    write_output(cand_dict, out_dir / "candidate_pairs.tsv", "candidate_entity_ids")
    write_output(match_dict, out_dir / "matching_results.tsv", "matched_entity_ids")
    
    print("Done! Cleaning up temp files.")
    os.remove(temp_cand_file)
    os.remove(temp_match_file)

if __name__ == "__main__":
    main()
