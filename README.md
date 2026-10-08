# Amazon ML Challenge 2026: Large-Scale Business Entity Resolution

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Macro F0.5](https://img.shields.io/badge/Val%20Macro%20F0.5-0.9698-brightgreen.svg)]()

An end-to-end, high-performance, and memory-safe **Business Entity Resolution (ER)** system designed for the Amazon ML Challenge 2026. This solution matches **1,732,544 Source-1 reference entities** against **9,969,589 candidate records** across Source 2 and Source 3 covering **France, the United States, and India**.

---

## 📌 Problem Overview

In large-scale commercial platforms like Amazon, business identity data arrives from multiple independent sources, each contributing partial, noisy, and fragmented records. The objective is to identify and resolve which records refer to the same real-world business entity.

### Key Constraints & Properties:
* **Reference Source:** **Source 1** is the deduplicated reference ground truth.
* **Match Cardinality:** Each Source-1 entity may match zero, one, or many records from Source 2 and Source 3 (**1-to-Many**).
* **Multi-Country Coverage:** The training set covers **US** and **India**, while the test set introduces a third country, **France**, that never appeared in training.
* **Scale:** $1.73\text{M}$ Source 1 entities $\times$ $9.97\text{M}$ Source 2/3 entities $\approx 17.2\text{ Trillion}$ potential pairwise comparisons.
* **Grading & Candidate Audit:** Organizers grade both the final matching predictions (`matching_results.tsv`) and the blocking efficiency (`candidate_pairs.tsv`). Solutions with unindexed massive candidate dumps ($>1,000$ candidates/entity) or metric gaming are penalized.

---

## 🏗️ System Architecture & Methodology

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          1. TEXT NORMALIZATION                          │
│   • Lowercase & URL Stripping   • Legal Entity Expansion (pvt, ltd, inc)│
│   • Street/Address Expansion    • Non-alphanumeric Character Cleaning   │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
┌────────────────────────────────────▼────────────────────────────────────┐
│                  2. SCALABLE BLOCKING & CANDIDATE POOL                  │
│   • Country Partitioning (France / US / India)                          │
│   • Inverted Posting Index on Informative Tokens (posting cap <= 2000)  │
│   • Candidate Selection: Floor >= 0.22, Top-15 Cutoff                   │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
┌────────────────────────────────────▼────────────────────────────────────┐
│            3. CONTINUOUS SUBLINEAR TF-IDF SIMILARITY SCORING            │
│   • Precomputed IDF Weights: idf(t) = log((N+1)/(df+1)) + 1.0           │
│   • Cosine Similarity Dot Product over Joint Name + Address Tokens      │
│   • Decision Thresholding (Match Threshold >= 0.46)                     │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
┌────────────────────────────────────▼────────────────────────────────────┐
│            4. GREEDY 1-TO-MANY REUSE-CAP COLLISION RESOLUTION           │
│   • Assigns each S2/S3 record strictly to its highest-scoring S1 entity │
│   • Eliminates cross-entity collision false positives                   │
│   • Strict Subset Invariant: Matches ⊆ Candidate Pairs (100% verified)  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Key Technical Innovations

### 1. Inverted Index Blocking with Compact Footprint
To eliminate brute-force matrix multiplication across trillions of pairs, the pipeline indexes all informative tokens from Source 2 and Source 3. Frequent boilerplate words are capped at $\le 2000$ postings. This limits pairwise comparisons to an average of **$14.1$ candidate pairs per entity** while retaining $>98\%$ candidate recall ceiling.

### 2. Unified Sublinear TF-IDF Text Representation
Rather than splitting name and address into fragile heuristic rules that break on irregular formatting (such as *"Michigan City, IN"* or suite numbers), the model normalizes the full entity string context into sublinear TF-IDF space:
$$\text{Sim}(S_1, S_2) = \frac{\sum_{t \in S_1 \cap S_2} \text{IDF}(t)^2}{\|S_1\| \cdot \|S_2\|}$$

### 3. Greedy 1-to-Many Collision Resolution (Reuse-Cap)
Because Source 1 is deduplicated, no single Source 2 or Source 3 record can belong to two distinct Source 1 businesses. The greedy reuse cap resolves multi-claim collisions by assigning the record strictly to the pair with the maximum cosine similarity score.

### 4. Zero-Lag Streaming Memory Architecture
The pipeline is engineered to run within **$< 400\text{ MB}$ RAM** using stream chunking and sparse data structures. CPU thread count is restricted to prevent system freezes or thermal throttling.

---

## 📊 Evaluation & Benchmark Results

### Macro $F_{0.5}$ Metric Formulation
The competition is evaluated using **Macro $F_{0.5}$**, weighting precision $4\times$ higher than recall ($\beta = 0.5$):
$$F_{0.5} = \frac{(1 + 0.5^2) \cdot \text{Precision} \cdot \text{Recall}}{0.5^2 \cdot \text{Precision} + \text{Recall}} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$$

### Validation & Test Set Metrics:
| Metric | Value / Specification |
| :--- | :---: |
| **Validation Precision** | **`96.1%`** |
| **Validation Recall** | **`94.2%`** |
| **Validation Macro $F_{0.5}$** | **`0.9698`** |
| **Total Test Entities Evaluated** | **`1,732,544`** ($100\%$) |
| **Total Matched Test Entities** | **`1,638,837`** ($94.59\%$, mirrors ground truth $94.42\%$) |
| **Singletons (Unmatched)** | **`93,707`** ($5.41\%$) |
| **France Partition Match Rate** | **`96.37%`** ($250,030 / 259,452$) |
| **US Partition Match Rate** | **`96.04%`** ($636,867 / 663,106$) |
| **India Partition Match Rate** | **`92.83%`** ($751,940 / 809,986$) |
| **Subset Invariant Violations** | **`0` (100% PASS)** |

---

## 📂 Repository Structure

```
amazon-ML-Challenge-2026/
│
├── dataset_small/                      # Lightweight standalone dataset for quick testing & reproduction
│   ├── train/                          # 3,000 S1 sample + ground truth + S2/S3 records
│   └── test/                           # S1/S2/S3 sample test splits
│
├── code/
│   └── business_entity_resolution/
│       ├── dataset/utils/              # Submission format & ID validator
│       └── src/                        # EntiLink pipeline & matching modules
│
├── output/
│   ├── matching_results.zip            # Compressed final matching results TSV (50 MB)
│   ├── candidate_pairs_part1.zip       # Multi-part compressed candidate pairs TSV
│   ├── candidate_pairs_part2.zip
│   └── candidate_pairs_part3.zip
│
├── generate_final_fast_submission.py   # Complete, low-memory production submission generator
├── generate_full_submission.py         # Full-scale TF-IDF matching generator
├── score_predictions.py                # Macro F0.5 local validation evaluation tool
├── Documentation.md                    # Detailed architectural notes & candidate tradeoff analysis
├── .gitignore                          # Git configuration excluding multi-GB raw dataset zips
└── README.md                           # Comprehensive documentation & walkthrough
```

---

## 🚀 Quickstart & Reproduction Guide

### 1. Prerequisites
Install required standard dependencies:
```bash
pip install pandas numpy scikit-learn scipy
```

### 2. Run Local Evaluation on Validation Split
```bash
python generate_final_fast_submission.py
```

### 3. Validate Submission Format & Invariants
```bash
python code/business_entity_resolution/dataset/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir code/business_entity_resolution/dataset/test
```

### 4. Extract Compressed Results
To extract the complete generated TSV files from the repository:
```python
import zipfile

# Extract matching_results.tsv
with zipfile.ZipFile("output/matching_results.zip", "r") as z:
    z.extractall("output")

# Combine and extract candidate_pairs.tsv
with open("output/candidate_pairs.tsv.zip", "wb") as f_out:
    for part in ["output/candidate_pairs_part1.zip", "output/candidate_pairs_part2.zip", "output/candidate_pairs_part3.zip"]:
        with open(part, "rb") as f_in:
            f_out.write(f_in.read())

with zipfile.ZipFile("output/candidate_pairs.tsv.zip", "r") as z:
    z.extractall("output")
```

---

## 👥 Authors
* **Mohit Sharma** ([@mohitsharmaa21](https://github.com/mohitsharmaa21))
* Developed for **Amazon ML Challenge 2026**
