# Amazon ML Challenge 2026 — Entity Resolution Pipeline

## Overview
This is a complete, end-to-end Entity Resolution pipeline for the Amazon ML Challenge 2026.  
It resolves ~24.2 million business records across 3 sources (US, India, France) using:
- **Multi-view text normalization** (Unicode, transliteration, accent folding, legal suffix extraction)
- **Sparse TF-IDF retrieval** with `sparse_dot_topn` (char n-grams, rare tokens, transliterated names)
- **28 pairwise features** (Jaro-Winkler, Jaccard, token overlap, address similarity, provenance bits)
- **LightGBM classifier** with calibration and macro F₀.₅ threshold tuning
- **SQLite-backed storage** with integer IDs throughout (no string/int mismatch bugs)

Target metric: **Macro F₀.₅ ≥ 0.99**

---

## Quick Start on AWS SageMaker

### 1. Create a Notebook Instance
1. Go to **AWS Console** → Search **Amazon SageMaker AI** → Click **Notebooks** → **Notebook instances**
2. Click **Create notebook instance**
   - **Name:** `amazon-ml-pipeline`
   - **Instance type:** `ml.m5.2xlarge` (8 cores, 32GB RAM) or larger if available
   - **Volume size:** `150 GB` ← **Critical! Do not skip this.**
   - **IAM role:** Create a new role → Any S3 bucket → Create
3. Click **Create**, wait for "InService", then click **Open JupyterLab**

### 2. Upload Files
Upload these files to the `SageMaker/` directory in JupyterLab:
- `er.py` (the pipeline)
- Your dataset folder as `dataset.zip`

### 3. Setup Environment
Open a **Terminal** in JupyterLab (File → New → Terminal) and run:

```bash
cd SageMaker

# Unzip the dataset
unzip -q dataset.zip

# Create Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify the Pipeline
```bash
python er.py --self-test
```
Expected output: `Core regression tests passed`

### 5. Run the Full Pipeline
```bash
# Use nohup so it survives browser disconnection
nohup python er.py --train dataset/train \
                   --test dataset/test \
                   --work run_full \
                   --output final_submission > run_log.txt 2>&1 &
```

You can now **close your browser**. The pipeline runs in the background (~6-10 hours on m5.2xlarge).

### 6. Check Progress & Download Results
```bash
# Check if it's still running
ps aux | grep er.py

# View the latest output
tail -50 run_log.txt
```

When finished, download from `final_submission/`:
- `matching_results.tsv` — Your submission file
- `candidate_pairs.tsv` — Candidate audit file

### 7. Validate & Submit
```bash
source .venv/bin/activate
python student_resource/utils/validate_submission.py \
  --matching final_submission/matching_results.tsv \
  --candidate final_submission/candidate_pairs.tsv \
  --test-dir dataset/test --check-ids
```

### 8. STOP YOUR INSTANCE!
⚠️ **Go to AWS Console → SageMaker → Notebook instances → Select yours → STOP**  
The m5.2xlarge costs ~$0.46/hr. Don't leave it running overnight after the job finishes!

---

## Dataset Structure
Place your dataset files like this:
```
dataset/
├── train/
│   ├── train_source1.tsv    (2.2M rows — S1, clean English)
│   ├── train_source2.tsv    (5.0M rows — S2, noisy, multilingual)
│   ├── train_source3.tsv    (5.3M rows — S3, noisy, multilingual)
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv     (~1.7M rows)
    ├── test_source2.tsv     (~4.9M rows)
    └── test_source3.tsv     (~5.1M rows)
```

## Architecture
The pipeline runs as a single `er.py` module with 20 sections:

```
Ingest TSVs → SQLite DB
       ↓
Multi-view Normalization (raw, unicode, transliterated, accent-folded, legal-stripped)
       ↓
Build Sparse TF-IDF Indexes (char n-grams, rare tokens, transliterated names)
       ↓
Candidate Retrieval (sparse_dot_topn top-k per channel)
       ↓
Reciprocal Rank Fusion (merge channels, cap at k=60)
       ↓
28 Pairwise Features (string similarity, address, provenance)
       ↓
LightGBM Classifier (trained on FIT split, early-stopped on STOP split)
       ↓
Calibration (Platt vs Isotonic, selected on CAL_SELECT)
       ↓
Threshold Tuning (macro F₀.₅ on TUNE split)
       ↓
Final Evaluation (frozen on FINAL split)
       ↓
Test Inference → matching_results.tsv + candidate_pairs.tsv
```

## Key Design Decisions
- **No dense embeddings / FAISS** — Pure sparse retrieval fits in CPU RAM
- **No libpostal** — Character n-gram Jaccard on raw addresses works and avoids 2GB memory + C build issues
- **SQLite with integer IDs** — No string/int key mismatch (fixes confirmed Bug B2)
- **`<NULL>` cleaned BEFORE generic null removal** — Fixes confirmed Bug B5
- **Connected-component GroupKFold splitting** — No label leakage between train/val/test
- **Separate calibration and threshold data** — Fixes confirmed Bug B8

## Cost Estimate
| Instance | Cost/hr | Est. Runtime | Total Cost |
|---|---|---|---|
| ml.m5.2xlarge | $0.46 | 6-10 hrs | ~$5 |
| ml.m5.4xlarge | $0.92 | 3-5 hrs | ~$4 |
| ml.m5.12xlarge | $2.76 | 1-2 hrs | ~$4 |

## License
MIT
