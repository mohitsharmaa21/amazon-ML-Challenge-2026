r"""
audit_reused_matches.py — flag Source-2/3 IDs that appear as a "match" for
MULTIPLE different Source-1 entities in a matching_results.tsv file.

Since Source 1 is explicitly the deduplicated reference source (each S1
entity is a distinct real business), a single S2/S3 record legitimately
matching many different S1 entities at once is very unlikely -- if it
happens a lot, that's strong independent evidence of classifier
overmatching, regardless of whether the business name is short or long.

This is a more principled diagnostic than a name-length heuristic: it
doesn't need any assumption about what makes a name "uninformative" --
it directly measures a structural impossibility (one real business
can't be the true match for many distinct other businesses).

Usage:
    python3 audit_reused_matches.py --matching output_run\matching_results.tsv --min-reuse 2
"""

import argparse
from collections import defaultdict

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", required=True)
    ap.add_argument("--min-reuse", type=int, default=2,
                     help="flag any S2/S3 ID matched to at least this many different S1 entities")
    args = ap.parse_args()

    df = pd.read_csv(args.matching, sep="\t", dtype=str).fillna("")

    reused_by_id = defaultdict(list)
    for row in df.itertuples(index=False):
        ids_str = str(row.matched_entity_ids).strip()
        if not ids_str:
            continue
        for mid in ids_str.split(","):
            reused_by_id[mid].append(row.source1_entity_id)

    flagged = {mid: s1_list for mid, s1_list in reused_by_id.items() if len(s1_list) >= args.min_reuse}

    if not flagged:
        print(f"No S2/S3 ID is matched to {args.min_reuse}+ different S1 entities. Looks clean.")
        return

    print(f"Found {len(flagged)} S2/S3 IDs each matched to {args.min_reuse}+ different S1 entities "
          f"(near-impossible if Source 1 is genuinely deduplicated):\n")
    for mid, s1_list in sorted(flagged.items(), key=lambda kv: -len(kv[1])):
        print(f"  {mid} -- matched to {len(s1_list)} different S1 entities: {s1_list}")

    total_suspect_pairs = sum(len(s1_list) for s1_list in flagged.values())
    print(f"\nTotal suspect (S1, matched_id) pairs involved: {total_suspect_pairs}")
    print("Recommendation: at most ONE of each flagged S2/S3 ID's matches can be correct -- "
          "the rest are near-certainly false positives worth removing.")


if __name__ == "__main__":
    main()