import os
import sys
import time
import gc
import re
from collections import defaultdict

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
    "house", "studio", "works", "cafe", "bar", "restaurant", "hotel",
    "societe", "saint", "st", "de", "la", "le", "du", "des", "les",
    "en", "un", "une", "par", "pour", "sur", "dans"
}

ALL_STOPWORDS = LEGAL_STOPWORDS | GENERIC_WORDS

RE_URL1 = re.compile(r"www\.[a-z0-9\-]+\.(com|org|net|in|fr|co|io)")
RE_URL2 = re.compile(r"\.(com|org|net|in|fr|io)\b")
RE_NON_ALPHANUM = re.compile(r"[^\w\s]")
RE_SPACES = re.compile(r"\s+")
RE_NUMS = re.compile(r"\b0*([1-9][0-9]*)\b")
RE_ALPHA_WORDS = re.compile(r"\b[a-z]{2,20}\b")

ABBREV_RULES = [
    (re.compile(r"\bcorp\b"), "corporation"),
    (re.compile(r"\bco\b"), "company"),
    (re.compile(r"\binc\b"), "incorporated"),
    (re.compile(r"\bltd\b"), "limited"),
    (re.compile(r"\bpvt\b"), "private"),
    (re.compile(r"\bllp\b"), "limited liability partnership"),
    (re.compile(r"\bllc\b"), "limited liability company"),
    (re.compile(r"\brd\b"), "road"),
    (re.compile(r"\bst\b"), "street"),
    (re.compile(r"\bave\b"), "avenue"),
    (re.compile(r"\bblvd\b"), "boulevard"),
    (re.compile(r"\bapt\b"), "apartment"),
    (re.compile(r"\bfl\b"), "floor"),
    (re.compile(r"\bno\b"), "number"),
    (re.compile(r"\bnr\b"), "near"),
    (re.compile(r"\bopp\b"), "opposite"),
    (re.compile(r"\bh\.?no\.?\b"), "number"),
    (re.compile(r"\bhno\b"), "number"),
    (re.compile(r"\bste\b"), "suite"),
    (re.compile(r"\bdr\b"), "drive"),
    (re.compile(r"\bln\b"), "lane"),
    (re.compile(r"\bhwy\b"), "highway"),
    (re.compile(r"\bpkwy\b"), "parkway"),
    (re.compile(r"\bct\b"), "court"),
    (re.compile(r"\bpl\b"), "place"),
    (re.compile(r"\bsq\b"), "square"),
    (re.compile(r"\bter\b"), "terrace"),
]

def clean_str(s):
    if not s:
        return ""
    s = str(s).lower()
    if "&" in s:
        s = s.replace("&", " and ")
    s = RE_URL1.sub(" ", s)
    s = RE_URL2.sub(" ", s)
    s = RE_NON_ALPHANUM.sub(" ", s)
    s = RE_SPACES.sub(" ", s).strip()
    for pat, repl in ABBREV_RULES:
        s = pat.sub(repl, s)
    return s

def strip_all_spaces_and_legal(s):
    tokens = [w for w in s.split() if w not in LEGAL_STOPWORDS]
    return "".join(tokens)

def extract_numbers(s):
    return tuple(RE_NUMS.findall(s))

def extract_state(addr, country):
    if country == "us":
        for name, code in US_STATE_NAMES.items():
            if name in addr:
                return code
        for t in set(RE_ALPHA_WORDS.findall(addr)):
            if t in US_STATES:
                return t
    elif country == "india":
        for name, code in INDIA_STATE_NAMES.items():
            if name in addr:
                return code
        for t in set(RE_ALPHA_WORDS.findall(addr)):
            if t in INDIA_STATES and t not in {"as", "in", "or", "is", "to", "at", "no", "dr"}:
                return t
    return ""

def get_sig_tokens(s):
    return [w for w in s.split() if w not in ALL_STOPWORDS and len(w) > 1]

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

def score_pair_fast(r1_parsed, r2_tuple):
    state1 = r1_parsed[5]
    state2 = r2_tuple[5]
    if state1 and state2 and state1 != state2:
        return -1.0
        
    name2 = r2_tuple[1]
    core2 = r2_tuple[2]
    addr2 = r2_tuple[3]
    street2 = r2_tuple[4]
    
    all_toks2 = set(name2.split())
    sig_toks2 = set(get_sig_tokens(name2))
    grams2 = get_char_3grams(name2)
    street_toks2 = set(street2.split())
    addr_toks2 = set(addr2.split())
    
    sig_jacc = jaccard(r1_parsed[9], sig_toks2)
    all_name_jacc = jaccard(r1_parsed[8], all_toks2)
    gram_jacc = jaccard(r1_parsed[10], grams2)
    
    exact_core = 1.0 if r1_parsed[2] and r1_parsed[2] == core2 and len(core2) >= 3 else 0.0
    core_substr = 1.0 if r1_parsed[2] and core2 and (r1_parsed[2] in core2 or core2 in r1_parsed[2]) and min(len(r1_parsed[2]), len(core2)) >= 4 else 0.0
    
    street_jacc = jaccard(r1_parsed[11], street_toks2)
    addr_jacc = jaccard(r1_parsed[12], addr_toks2)
    
    num_bonus = 0.0
    s_nums1 = r1_parsed[6]
    s_nums2 = r2_tuple[6]
    if s_nums1 and s_nums2:
        if set(s_nums1) & set(s_nums2):
            num_bonus = 0.20
        else:
            num_bonus = -0.25
    elif r1_parsed[7] and r2_tuple[7]:
        if set(r1_parsed[7]) & set(r2_tuple[7]):
            num_bonus = 0.10
            
    name_score = 0.35 * sig_jacc + 0.25 * all_name_jacc + 0.20 * gram_jacc + 0.20 * max(exact_core, 0.7 * core_substr)
    addr_score = 0.65 * street_jacc + 0.35 * addr_jacc
    
    state_bonus = 0.10 if (state1 and state2 and state1 == state2) else 0.0
    
    total_score = 0.58 * name_score + 0.38 * addr_score + num_bonus + state_bonus
    if exact_core and (street_jacc > 0 or addr_jacc > 0 or num_bonus > 0 or state_bonus > 0):
        total_score = max(total_score, 0.75 + 0.25 * addr_score)
        
    return total_score

def read_source_file_by_country(filepath, target_country):
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

MAX_POSTING_LEN = 300

# CONFIG C4 PARAMETERS
CANDIDATE_FLOOR = 0.25
CANDIDATE_TOP_K = 15
MATCH_THRESHOLD = 0.49
MIN_NAME_LEN = 4

def process_country(country, test_dir):
    print(f"\n=================================================================")
    print(f"--> PROCESSING COUNTRY: {country.upper()}")
    print(f"=================================================================")
    t0 = time.time()
    
    # 1. Fast read S1
    s1_raw = read_source_file_by_country(os.path.join(test_dir, "test_source1.tsv"), country)
    print(f"Loaded {len(s1_raw)} Source 1 records for {country} in {time.time()-t0:.2f}s.")
    
    s1_records = []
    for eid, bname, baddr in s1_raw:
        c_name = clean_str(bname)
        c_core = strip_all_spaces_and_legal(c_name)
        c_addr = clean_str(baddr)
        c_street = c_addr.split(",")[0] if c_addr else ""
        s1_records.append((
            eid, c_name, c_core, c_addr, c_street,
            extract_state(c_addr, country),
            extract_numbers(c_street),
            extract_numbers(c_addr),
            set(c_name.split()),
            set(get_sig_tokens(c_name)),
            get_char_3grams(c_name),
            set(c_street.split()),
            set(c_addr.split())
        ))
    del s1_raw
    
    # 2. Fast read S2 and S3 and build index
    t_idx = time.time()
    s2_raw = read_source_file_by_country(os.path.join(test_dir, "test_source2.tsv"), country)
    s3_raw = read_source_file_by_country(os.path.join(test_dir, "test_source3.tsv"), country)
    s23_raw = s2_raw + s3_raw
    del s2_raw, s3_raw
    print(f"Loaded {len(s23_raw)} S2/S3 records in {time.time()-t_idx:.2f}s. Indexing...")
    
    s23_records = []
    s23_token_index = defaultdict(list)
    s23_core_name_index = defaultdict(list)
    s23_first_token_index = defaultdict(list)
    
    for idx, (eid, bname, baddr) in enumerate(s23_raw):
        c_name = clean_str(bname)
        c_core = strip_all_spaces_and_legal(c_name)
        c_addr = clean_str(baddr)
        c_street = c_addr.split(",")[0] if c_addr else ""
        
        rec = (
            eid, c_name, c_core, c_addr, c_street,
            extract_state(c_addr, country),
            extract_numbers(c_street),
            extract_numbers(c_addr)
        )
        s23_records.append(rec)
        
        sig_toks = get_sig_tokens(c_name)
        all_toks = c_name.split()
        
        if c_core and len(c_core) >= 3:
            s23_core_name_index[c_core].append(idx)
        for tok in sig_toks:
            if len(tok) >= 3:
                s23_token_index[tok].append(idx)
        if all_toks and len(all_toks[0]) >= 3:
            s23_first_token_index[all_toks[0]].append(idx)
            
    del s23_raw
    
    s23_token_index = {k: v for k, v in s23_token_index.items() if len(v) <= MAX_POSTING_LEN}
    s23_first_token_index = {k: v for k, v in s23_first_token_index.items() if len(v) <= MAX_POSTING_LEN}
    s23_core_name_index = {k: v for k, v in s23_core_name_index.items() if len(v) <= MAX_POSTING_LEN}
    
    print(f"Indexed {len(s23_records)} S23 records for {country} in {time.time()-t_idx:.2f}s.")
    
    # 3. Match S1 records with Config C4 (Floor >= 0.25 + Top-15)
    t_match = time.time()
    candidates_by_s1 = {}
    matches_with_scores = []
    
    for idx, r1 in enumerate(s1_records):
        eid = r1[0]
        c_core = r1[2]
        sig_toks = r1[9]
        all_toks = r1[8]
        
        cand_indices = set()
        if c_core and len(c_core) >= 3:
            cand_indices.update(s23_core_name_index.get(c_core, []))
        for tok in sig_toks:
            if len(tok) >= 3:
                cand_indices.update(s23_token_index.get(tok, []))
        if all_toks:
            first_w = next(iter(all_toks))
            if len(first_w) >= 3:
                cand_indices.update(s23_first_token_index.get(first_w, []))
                
        # Score all candidate pairs
        scored_pairs = []
        for c_idx in cand_indices:
            r2 = s23_records[c_idx]
            score = score_pair_fast(r1, r2)
            if score >= CANDIDATE_FLOOR:
                scored_pairs.append((r2[0], score, len(r1[1]), len(r2[1])))
                
        # Sort descending by score
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Apply Top-K cap for candidate_pairs.tsv (Config C4)
        top_k_candidates = scored_pairs[:CANDIDATE_TOP_K]
        
        # Candidate IDs list for candidate_pairs.tsv
        cand_ids = [c[0] for c in top_k_candidates]
        candidates_by_s1[eid] = cand_ids
        
        # Matches are subset of candidates with score >= MATCH_THRESHOLD (0.49)
        for s2_id, score, l1, l2 in top_k_candidates:
            if score >= MATCH_THRESHOLD:
                matches_with_scores.append((eid, s2_id, score, l1, l2))
                
        if (idx + 1) % 100000 == 0 or (idx + 1) == len(s1_records):
            print(f"  [{country.upper()}] Matched {idx+1}/{len(s1_records)} records ({time.time()-t_match:.1f}s)...")
            
    print(f"Scoring completed in {time.time()-t_match:.2f}s. Scored matches above threshold: {len(matches_with_scores)}")
    
    # 4. Reuse Cap for final matches
    print(f"Applying Reuse-Cap for {country}...")
    best_for_s2 = {}
    for s1_id, s2_id, score, l1, l2 in matches_with_scores:
        if MIN_NAME_LEN > 0 and (l1 < MIN_NAME_LEN or l2 < MIN_NAME_LEN):
            continue
        if s2_id not in best_for_s2 or score > best_for_s2[s2_id][0]:
            best_for_s2[s2_id] = (score, s1_id)
            
    final_matches_by_s1 = defaultdict(set)
    for s1_id, s2_id, score, l1, l2 in matches_with_scores:
        if MIN_NAME_LEN > 0 and (l1 < MIN_NAME_LEN or l2 < MIN_NAME_LEN):
            continue
        if s2_id in best_for_s2 and best_for_s2[s2_id][1] == s1_id:
            final_matches_by_s1[s1_id].add(s2_id)
            
    print(f"Done {country.upper()}: {len(final_matches_by_s1)} entities with matches in {time.time()-t0:.2f}s total.")
    
    del s1_records, s23_records, s23_token_index, s23_core_name_index, s23_first_token_index, matches_with_scores
    gc.collect()
    
    return final_matches_by_s1, candidates_by_s1

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
        
    print("Reading required Source 1 IDs in exact order...")
    all_s1_ids = []
    with open(os.path.join(test_dir, "test_source1.tsv"), "r", encoding="utf-8", errors="replace") as f:
        f.readline()
        for line in f:
            parts = line.split("\t", 1)
            if parts and parts[0].strip():
                all_s1_ids.append(parts[0].strip())
    print(f"Total required Test Source 1 entities: {len(all_s1_ids)}")
    
    all_final_matches = {}
    all_candidates = {}
    
    for country in ["france", "us", "india"]:
        matches, cands = process_country(country, test_dir)
        all_final_matches.update(matches)
        all_candidates.update(cands)
        
    print(f"\n=======================================================")
    print(f"WRITING FINAL SUBMISSION FILES TO DISK...")
    print(f"=======================================================")
    
    for out_dir in out_dirs:
        m_path = os.path.join(out_dir, "matching_results.tsv")
        c_path = os.path.join(out_dir, "candidate_pairs.tsv")
        
        print(f"Writing {m_path}...")
        with open(m_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            for s1_id in all_s1_ids:
                matched = sorted(all_final_matches.get(s1_id, set()))
                f.write(f"{s1_id}\t{','.join(matched)}\n")
                
        print(f"Writing {c_path}...")
        with open(c_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            for s1_id in all_s1_ids:
                cands = all_candidates.get(s1_id, [])
                f.write(f"{s1_id}\t{','.join(cands)}\n")

    print(f"\n=======================================================")
    print(f"COMPLETE SUCCESS! Runtime: {time.time()-t_start:.2f}s ({(time.time()-t_start)/60:.1f} minutes).")
    print(f"=======================================================")

if __name__ == "__main__":
    main()
