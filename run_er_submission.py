import os
import sys
import time
import re
import unicodedata
from collections import defaultdict
import numpy as np
import pandas as pd
import argparse

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa",
    "ks": "kansas", "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi", "mo": "missouri",
    "mt": "montana", "ne": "nebraska", "nv": "nevada", "nh": "new hampshire", "nj": "new jersey",
    "nm": "new mexico", "ny": "new york", "nc": "north carolina", "nd": "north dakota", "oh": "ohio",
    "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina",
    "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah", "vt": "vermont",
    "va": "virginia", "wa": "washington", "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming",
    "dc": "district of columbia"
}
US_STATE_NAMES = {v: k for k, v in US_STATES.items()}

INDIA_STATES = {
    "ap": "andhra pradesh", "ar": "arunachal pradesh", "as": "assam", "br": "bihar", "cg": "chhattisgarh",
    "ga": "goa", "gj": "gujarat", "hr": "haryana", "hp": "himachal pradesh", "jh": "jharkhand",
    "ka": "karnataka", "kl": "kerala", "mp": "madhya pradesh", "mh": "maharashtra", "mn": "manipur",
    "ml": "meghalaya", "mz": "mizoram", "nl": "nagaland", "or": "odisha", "pb": "punjab",
    "rj": "rajasthan", "sk": "sikkim", "tn": "tamil nadu", "ts": "telangana", "tr": "tripura",
    "up": "uttar pradesh", "uk": "uttarakhand", "wb": "west bengal", "dl": "delhi", "ch": "chandigarh",
    "py": "puducherry", "jk": "jammu and kashmir", "la": "ladakh"
}
INDIA_STATE_NAMES = {v: k for k, v in INDIA_STATES.items()}

LEGAL_STOPWORDS = {
    "inc", "incorporated", "corp", "corporation", "co", "company",
    "llc", "ltd", "limited", "pvt", "private", "llp", "pc", "pllc",
    "gmbh", "sarl", "sa", "sas", "bv", "nv", "ag", "sl", "srl",
    "lp", "lllp", "holdings", "holding", "group", "enterprises",
    "enterprise", "services", "service", "solutions", "solution",
    "technologies", "technology", "tech", "systems", "system",
    "international", "consulting", "consultancy", "ventures", "venture",
    "management", "associates", "industries", "industry"
}

GENERIC_WORDS = {
    "the", "and", "of", "in", "for", "on", "at", "to", "a", "an",
    "center", "centre", "store", "shop", "market", "mart", "club",
    "house", "studio", "works", "cafe", "bar", "restaurant", "hotel"
}

ALL_STOPWORDS = LEGAL_STOPWORDS | GENERIC_WORDS

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
    r"\bh\.?no\.?\b": "number",
    r"\bhno\b": "number",
    r"\bste\b": "suite",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bhwy\b": "highway",
    r"\bpkwy\b": "parkway",
    r"\bct\b": "court",
    r"\bpl\b": "place",
    r"\bsq\b": "square",
    r"\bter\b": "terrace",
}

def remove_accents(input_str):
    if not input_str:
        return ""
    nfkd = unicodedata.normalize('NFKD', str(input_str))
    return "".join([c for c in nfkd if not unicodedata.combining(c)])

def clean_str(s):
    if not s or pd.isna(s):
        return ""
    s = remove_accents(str(s).lower())
    s = re.sub(r"www\.[a-z0-9\-]+\.(com|org|net|in|fr|co|io)", " ", s)
    s = re.sub(r"\.(com|org|net|in|fr|io)\b", " ", s)
    s = s.replace("&", " and ")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for pattern, repl in ABBREV_MAP.items():
        s = re.sub(pattern, repl, s)
    return re.sub(r"\s+", " ", s).strip()

def strip_all_spaces_and_legal(s):
    s = clean_str(s)
    tokens = [w for w in s.split() if w not in LEGAL_STOPWORDS]
    return "".join(tokens)

def extract_numbers(s):
    nums = re.findall(r"\b0*([1-9][0-9]*)\b", str(s))
    return set(nums)

def extract_state(addr, country):
    addr = str(addr).lower()
    tokens = set(re.findall(r"\b[a-z]{2,20}\b", addr))
    if country == "us":
        for name, code in US_STATE_NAMES.items():
            if name in addr:
                return code
        for t in tokens:
            if t in US_STATES:
                return t
    elif country == "india":
        for name, code in INDIA_STATE_NAMES.items():
            if name in addr:
                return code
        for t in tokens:
            if t in INDIA_STATES and t not in {"as", "in", "or", "is", "to", "at", "no", "dr"}:
                return t
    return None

def get_tokens(s):
    return [w for w in s.split() if len(w) > 0]

def get_sig_tokens(s):
    return [w for w in get_tokens(s) if w not in ALL_STOPWORDS and len(w) > 1]

def get_char_3grams(s):
    s = f"^{s}$"
    if len(s) < 3:
        return {s}
    return {s[i:i+3] for i in range(len(s)-2)}

def jaccard(set1, set2):
    if not set1 or not set2:
        return 0.0
    inter = len(set1 & set2)
    return inter / (len(set1) + len(set2) - inter)

def compute_pair_score(rec1, rec2):
    # State check
    if rec1["state"] and rec2["state"] and rec1["state"] != rec2["state"]:
        return -1.0 # conflicting state
        
    sig_jacc = jaccard(rec1["name_sig_tokens"], rec2["name_sig_tokens"])
    all_name_jacc = jaccard(rec1["name_tokens"], rec2["name_tokens"])
    gram_jacc = jaccard(rec1["name_3grams"], rec2["name_3grams"])
    
    exact_core = 1.0 if rec1["core_name"] and rec1["core_name"] == rec2["core_name"] and len(rec1["core_name"]) >= 3 else 0.0
    core_substr = 1.0 if rec1["core_name"] and rec2["core_name"] and (rec1["core_name"] in rec2["core_name"] or rec2["core_name"] in rec1["core_name"]) and min(len(rec1["core_name"]), len(rec2["core_name"])) >= 4 else 0.0
    
    street_jacc = jaccard(rec1["street_tokens"], rec2["street_tokens"])
    addr_jacc = jaccard(rec1["addr_tokens"], rec2["addr_tokens"])
    
    num_bonus = 0.0
    if rec1["street_numbers"] and rec2["street_numbers"]:
        if rec1["street_numbers"] & rec2["street_numbers"]:
            num_bonus = 0.20
        else:
            num_bonus = -0.25
    elif rec1["addr_numbers"] and rec2["addr_numbers"]:
        if rec1["addr_numbers"] & rec2["addr_numbers"]:
            num_bonus = 0.10
            
    name_score = 0.35 * sig_jacc + 0.25 * all_name_jacc + 0.20 * gram_jacc + 0.20 * max(exact_core, 0.7 * core_substr)
    addr_score = 0.65 * street_jacc + 0.35 * addr_jacc
    
    state_bonus = 0.10 if (rec1["state"] and rec2["state"] and rec1["state"] == rec2["state"]) else 0.0
    
    total_score = 0.58 * name_score + 0.38 * addr_score + num_bonus + state_bonus
    if exact_core and (street_jacc > 0 or addr_jacc > 0 or num_bonus > 0 or state_bonus > 0):
        total_score = max(total_score, 0.75 + 0.25 * addr_score)
        
    return total_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", required=True, help="Directory containing test_source1.tsv, test_source2.tsv, test_source3.tsv")
    parser.add_argument("--out-dir", required=True, help="Output directory to save matching_results.tsv and candidate_pairs.tsv")
    parser.add_argument("--threshold", type=float, default=0.49, help="Matching score threshold (default: 0.49)")
    parser.add_argument("--min-name-len", type=int, default=4, help="Min name length filter (default: 4)")
    args = parser.parse_args()

    test_dir = args.test_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading test datasets from {test_dir}...")
    t0 = time.time()
    s1 = pd.read_csv(os.path.join(test_dir, "test_source1.tsv"), sep="\t", dtype=str).fillna("")
    s2 = pd.read_csv(os.path.join(test_dir, "test_source2.tsv"), sep="\t", dtype=str).fillna("")
    s3 = pd.read_csv(os.path.join(test_dir, "test_source3.tsv"), sep="\t", dtype=str).fillna("")
    print(f"Loaded: S1={len(s1)}, S2={len(s2)}, S3={len(s3)} in {time.time()-t0:.2f}s")

    # Combine S2 and S3
    s23 = pd.concat([s2, s3], ignore_index=True)

    print("Building Inverted Indices for S2/S3 candidate pools...")
    t_idx = time.time()
    s23_by_id = {}
    s23_token_index = defaultdict(list)
    s23_core_name_index = defaultdict(list)
    s23_first_token_index = defaultdict(list)

    for idx, row in enumerate(s23.itertuples(index=False)):
        c_name = clean_str(row.business_name)
        c_core = strip_all_spaces_and_legal(row.business_name)
        c_addr = clean_str(row.business_address)
        c_street = clean_str(str(row.business_address).split(",")[0] if row.business_address else "")
        c_country = str(row.country).lower().strip()
        
        rec = {
            "id": row.entity_id,
            "name": c_name,
            "core_name": c_core,
            "addr": c_addr,
            "street": c_street,
            "country": c_country,
            "state": extract_state(c_addr, c_country),
            "name_tokens": set(get_tokens(c_name)),
            "name_sig_tokens": set(get_sig_tokens(c_name)),
            "name_3grams": get_char_3grams(c_name),
            "street_tokens": set(get_tokens(c_street)),
            "addr_tokens": set(get_tokens(c_addr)),
            "street_numbers": extract_numbers(c_street),
            "addr_numbers": extract_numbers(c_addr),
        }
        s23_by_id[row.entity_id] = rec
        
        if c_core and len(c_core) >= 3:
            s23_core_name_index[(c_country, c_core)].append(row.entity_id)
        for tok in rec["name_sig_tokens"]:
            if len(tok) >= 3:
                s23_token_index[(c_country, tok)].append(row.entity_id)
        toks = get_tokens(c_name)
        if toks and len(toks[0]) >= 3:
            s23_first_token_index[(c_country, toks[0])].append(row.entity_id)

    print(f"Indexed {len(s23_by_id)} S23 records in {time.time()-t_idx:.2f}s.")

    print("Generating candidate pairs and scoring matches for Test Source 1 entities...")
    t_match = time.time()
    all_s1_ids = s1["entity_id"].tolist()
    candidates_by_s1 = defaultdict(set)
    all_matches_with_scores = [] # (s1_id, s2_id, score, len1, len2)

    for idx, row in enumerate(s1.itertuples(index=False)):
        s1_id = row.entity_id
        c_name = clean_str(row.business_name)
        c_core = strip_all_spaces_and_legal(row.business_name)
        c_addr = clean_str(row.business_address)
        c_street = clean_str(str(row.business_address).split(",")[0] if row.business_address else "")
        c_country = str(row.country).lower().strip()
        
        rec1 = {
            "id": s1_id,
            "name": c_name,
            "core_name": c_core,
            "addr": c_addr,
            "street": c_street,
            "country": c_country,
            "state": extract_state(c_addr, c_country),
            "name_tokens": set(get_tokens(c_name)),
            "name_sig_tokens": set(get_sig_tokens(c_name)),
            "name_3grams": get_char_3grams(c_name),
            "street_tokens": set(get_tokens(c_street)),
            "addr_tokens": set(get_tokens(c_addr)),
            "street_numbers": extract_numbers(c_street),
            "addr_numbers": extract_numbers(c_addr),
        }
        
        cand_ids = set()
        if c_core and len(c_core) >= 3:
            cand_ids.update(s23_core_name_index.get((c_country, c_core), []))
        for tok in rec1["name_sig_tokens"]:
            if len(tok) >= 3:
                cand_ids.update(s23_token_index.get((c_country, tok), []))
        toks = get_tokens(c_name)
        if toks and len(toks[0]) >= 3:
            cand_ids.update(s23_first_token_index.get((c_country, toks[0]), []))
            
        candidates_by_s1[s1_id] = cand_ids
        
        for c_id in cand_ids:
            rec2 = s23_by_id[c_id]
            score = compute_pair_score(rec1, rec2)
            if score >= args.threshold:
                all_matches_with_scores.append((s1_id, c_id, score, len(c_name), len(rec2["name"])))
                
        if (idx + 1) % 100000 == 0 or (idx + 1) == len(s1):
            print(f"  Processed {idx+1}/{len(s1)} Source 1 records ({time.time()-t_match:.1f}s)...")

    print(f"Candidate generation & scoring finished in {time.time()-t_match:.2f}s.")
    print(f"Total scored matches above threshold {args.threshold}: {len(all_matches_with_scores)}")

    # Apply Reuse Cap (One-to-Many Resolution)
    print("Applying global Reuse-Cap resolution...")
    best_for_s2 = {}
    for s1_id, s2_id, score, l1, l2 in all_matches_with_scores:
        if args.min_name_len > 0 and (l1 < args.min_name_len or l2 < args.min_name_len):
            continue
        if s2_id not in best_for_s2 or score > best_for_s2[s2_id][0]:
            best_for_s2[s2_id] = (score, s1_id)

    final_matches_by_s1 = defaultdict(set)
    for s1_id, s2_id, score, l1, l2 in all_matches_with_scores:
        if args.min_name_len > 0 and (l1 < args.min_name_len or l2 < args.min_name_len):
            continue
        if s2_id in best_for_s2 and best_for_s2[s2_id][1] == s1_id:
            final_matches_by_s1[s1_id].add(s2_id)

    # Write output files
    out_matching_path = os.path.join(out_dir, "matching_results.tsv")
    out_candidate_path = os.path.join(out_dir, "candidate_pairs.tsv")

    print(f"Writing {out_matching_path}...")
    with open(out_matching_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in all_s1_ids:
            matched = sorted(final_matches_by_s1.get(s1_id, set()))
            f.write(f"{s1_id}\t{','.join(matched)}\n")

    print(f"Writing {out_candidate_path}...")
    with open(out_candidate_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in all_s1_ids:
            cands = sorted(candidates_by_s1.get(s1_id, set()))
            f.write(f"{s1_id}\t{','.join(cands)}\n")

    print(f"\n=======================================================")
    print(f"SUCCESS! Submission files generated in {out_dir}:")
    print(f"  1. {out_matching_path}")
    print(f"  2. {out_candidate_path}")
    print(f"=======================================================")

if __name__ == "__main__":
    main()
