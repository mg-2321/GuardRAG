# GuardRAG: Aversarial Preference Training Pipeline Against Indirect Prompt Injection Attacks

A comprehensive framework for adversarial training and evaluating Retrieval-Augmented Generation (RAG) systems against In-Context Poisoning (IPI) attacks using Direct Preference Optimization (DPO).

## Overview

GuardRAG provides:
- **Poisoned Corpus Generation** - IPI corpus creation for multiple benchmark datasets
- **Multi-Retriever Evaluation** - BM25, Dense, SPLADE, and Hybrid retrievers
- **LLM Generation Testing** - Attack success rate and leakage detection with 8B and 70B models
- **DPO Defense Training** - Direct Preference Optimization for robust model training

## Research Overview: Two-Phase Framework

### Phase 1: RIPE-II - Retrieval-based IPI Evaluation Framework

**Paper:** [RIPE-II: Comprehensive Evaluation of In-Context Poisoning Attacks on RAG](https://drive.google.com/file/d/1HL38P9Vaa6xc7O_Gs1ch1ONSJ0v-_XyY/view?usp=sharing)

Phase 1 establishes the comprehensive evaluation framework for IPI attacks on RAG systems:

**Stage-by-Stage Evaluation:**
- **Stage 1-2**: Corpus & Query Preparation
- **Stage 3**: Retrieval evaluation (poison exposure in top-k)
- **Stage 4**: Context packing analysis
- **Stage 5**: Ranking and positioning effects
- **Stages 6**: Generation evaluation (ASR, leakage, refusal rates)

**Key Metrics:**
- Exposure Rate (ER@k): % queries with poisoned docs in top-k
- Ranking Drift Score (RDS): Position changes due to poisoning
- Attack Success Rate (ASR): % following attack directives
- Leakage Rate (LR): % containing injected content
- Over-Refusal Rate (ORR): False positives on clean data

**Datasets:** NFCorpus, FiQA, SciFact, HotpotQA, Natural Questions

### BEIR Benchmarks
| Dataset | Queries | Docs | Type |
|---------|---------|------|------|
| **NFCorpus** | 3,237 | 323,018 | Medical |
| **FiQA** | 6,648 | 57,638 | Finance |
| **SciFact** | 1,109 | 5,183 | Scientific |
| **HotpotQA** | 97,852 | 5,233,235 | Multi-hop |
| **Natural Questions** | 3,237 | 21,015,324 | Open-domain |

**Download:** https://github.com/beir-cellar/beir


### Phase 2: GuardRAG - Adversarial Defense Preference Training

Phase 2 focuses on building adversarial training pipeline on RIPE-II's evaluation framework to implement defense mechanisms against IPI attacks:

**Defense Mechanisms:**
- **Direct Preference Optimization (DPO)**: Train models to prefer clean over poisoned generations
- **SimPO**: Simplified preference optimization for robust RAG models
- **Preference Datasets**: Curated pairs of (clean, poisoned) generations for preference learning

**Adversarial Training Components:**
1. **Preference Data Generation**: Creates training pairs from Phase 1 evaluations
2. **DPO Training**: Optimizes models using preference learning
3. **Adversarial Evaluation**: Validates robustness against attacks
4. **Final Evaluation**: Comprehensive baseline vs DPO comparison with new metrics

**New Metrics (Phase 2.1):**
- **Helpfulness Score**: BLEU-based metric measuring response quality (0-1)
- **Naturalness Score**: Perplexity-based metric measuring language fluency (0-1)
- **Faithfulness Score**: Token overlap metric measuring context grounding (0-100%)


---

## Quick Start

### 1. Installation

```bash
# Install dependencies
bash complete_package_install.sh

# Verify installation
python main.py --help
```

### 2. Download Models & Data

```bash
# Download Llama 3.1 70B (optional, for generation evaluation)
bash download_llama70b.sh

# Download benchmark datasets (handled automatically)
python main.py --mode rag --corpus hotpotqa --sample 10
```

### 3. Run Phase 1: RIPE-II Evaluation

```bash
# Comparative retriever evaluation
python evaluation/run_evaluation.py --mode comparative --corpus hotpotqa --sample 100

# Stage-by-stage analysis
python evaluation/run_evaluation.py --mode stages --corpus fiqa

# Generation evaluation (Stages 6-7 - ASR/LR metrics)
python evaluation/run_evaluation.py --mode generation --corpus nfcorpus --sample 50
```

### 4. Generate Preference Training Data

```bash
# Generate preference pairs (clean vs poisoned responses)
# For a single corpus
python DPO/data/build_preference_pairs.py \
    --corpus nfcorpus \
    --output-dir data/preference/dpo_nfcorpus_full \
    --model mistralai/Mistral-7B-Instruct-v0.2

# For all corpuses (sequentially)
bash scripts/generate_dpo_all_corpuses_unified.sh

# Monitor generation progress
watch -n 60 'wc -l data/preference/dpo_*/*/clean.jsonl'
```

### 5. Train DPO Model

```bash
# Combine preference pairs from single corpus
cat data/preference/dpo_nfcorpus_full/nfcorpus/*.jsonl > \
    data/preference/dpo_nfcorpus_full/nfcorpus_combined.jsonl

# Train DPO model
python DPO/dpo/train_dpo.py \
    --model-name mistralai/Mistral-7B-Instruct-v0.2 \
    --train-file data/preference/dpo_nfcorpus_full/nfcorpus_combined.jsonl \
    --output-dir outputs/dpo_nfcorpus_mistral \
    --num-train-epochs 3 \
    --learning-rate 5e-5 \
    --per-device-train-batch-size 4 \
    --gradient-accumulation-steps 2 \
    --bf16
```

### 6. Run Final Evaluation (Baseline vs DPO)

```bash
# Run complete evaluation pipeline
bash scripts/run_final_evaluation.sh

# Or run with custom settings
python evaluation/final_evaluation.py \
    --baseline-model mistralai/Mistral-7B-Instruct-v0.2 \
    --dpo-model outputs/dpo_nfcorpus_mistral/final_model \
    --corpus nfcorpus \
    --output-dir evaluation/final_results

# Quick test (10 samples per dataset - 5 minutes)
python evaluation/final_evaluation.py \
    --dpo-model outputs/dpo_nfcorpus_mistral/final_model \
    --max-samples 10

# Analyze results
python evaluation/analyze_final_results.py \
    --results evaluation/final_results/detailed_results.json \
    --output-dir evaluation/final_results/analysis
```

### 7. Review Results

```bash
# View executive summary
cat evaluation/final_results/analysis/SUMMARY_REPORT.txt

# Export for Excel analysis
open evaluation/final_results/analysis/metrics_comparison.csv
```


**Format:** Each document as `{"_id": "...", "title": "...", "text": "..."}`

## Models & Resources

### LLMs
- **Llama 3.1 8B**: Local inference, ~15 min per corpus
  - Download: https://www.meta.com/llama/llama-downloads/
  
- **Llama 3.1 70B**: GPU required, ~3x slower than 8B
  - Download: https://www.meta.com/llama/llama-downloads/
  - Requires: 2x 40GB GPUs (A100) or equivalent

### Embedding Models
- **E5-Large** (dense): `intfloat/e5-large-v2` (1,024 dims)
- **SPLADE**: `naver/splade_pp_en_v1` (Sparse vectors)
- **BM25**: Built-in (no download needed)

### Reference Implementations
- **DPO Module**: `DPO/dpo/train_dpo.py`
- **SimPO Module**: `reference_models/simpo/`

## Architecture

```
GuardRAG/
├── evaluation/              # Phase 1 & 2: Evaluation
│   ├── run_evaluation.py    # Unified evaluator (comparative, stages, generation)
│   ├── metrics_calculator.py# Metric computations (ER@k, ASR, LR, ORR)
│   ├── stage_by_stage_evaluation.py  # Stage-by-stage analysis
│   ├── final_evaluation.py  # Phase 2.1: Baseline vs DPO comparison (NEW)
│   ├── analyze_final_results.py  # Results analyzer & visualizer (NEW)
│   └── diagnostic/          # Diagnostic tools
├── rag_pipeline/            # RAG implementation
│   ├── retrievers/          # BM25, Dense, SPLADE, Hybrid
│   ├── pipeline.py          # Main pipeline
│   └── generator.py         # LLM generation
├── DPO/                     # Phase 2: Adversarial Defense
│   ├── dpo/
│   │   └── train_dpo.py     # DPO training orchestrator
│   ├── data/
│   │   └── build_preference_pairs.py  # Preference pair generation
│   ├── reward_model/        # Reward modeling
│   └── prompting.py         # Prompt utilities
├── IPI_generators/          # Phase 1: Attack corpus generation
│   ├── ipi_*/               # Per-corpus poisoned data
│   └── dense_aligned_ipi_generator.py  # IPI generator
├── reference_models/        # Phase 2: Model implementations
│   ├── dpo/                 # DPO reference
│   └── simpo/               # SimPO reference
├── scripts/                 # Automation scripts
│   ├── run_final_evaluation.sh  # Phase 2.1: One-command evaluation (NEW)
│   ├── generate_dpo_all_corpuses_unified.sh  # Preference generation orchestrator
│   └── generate_dpo_*.sh    # Per-corpus generation scripts
├── FINAL_EVALUATION_GUIDE.md  # Phase 2.1: Complete evaluation guide (NEW)
├── EVALUATION_QUICK_REFERENCE.txt  # Quick reference (NEW)
├── EVALUATION_COMPLETION_CHECKLIST.txt  # Feature checklist (NEW)
├── main.py                  # CLI entry point
└── setup.py                 # Package configuration
```

## Commands Reference

### Installation & Setup
```bash
bash complete_package_install.sh          # Full installation
pip install -r requirements.txt           # Minimal install
python setup.py install                   # Package install
```

### Model Downloads
```bash
bash download_llama70b.sh                 # Download Llama 70B
```

### Phase 1: RIPE-II Evaluation
```bash
# Comparative retriever evaluation
python evaluation/run_evaluation.py --mode comparative --corpus hotpotqa

# Stage-by-stage pipeline analysis
python evaluation/run_evaluation.py --mode stages --corpus fiqa

# Generation evaluation (ASR/LR metrics)
python evaluation/run_evaluation.py --mode generation --corpus nfcorpus

# Batch evaluation on all corpora
python evaluation/run_evaluation.py --mode comparative --all --sample 100
```

### Phase 2: Preference Data Generation
```bash
# Generate preference pairs for single corpus
python DPO/data/build_preference_pairs.py \
    --corpus nfcorpus \
    --output-dir data/preference/dpo_nfcorpus_full \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --num-workers 8

# Generate for all corpuses (sequentially)
bash scripts/generate_dpo_all_corpuses_unified.sh

# Monitor progress
watch -n 60 'wc -l data/preference/dpo_*/*/clean.jsonl'
```

### Phase 2: DPO Model Training
```bash
# Combine preference pairs
cat data/preference/dpo_nfcorpus_full/nfcorpus/*.jsonl > \
    data/preference/dpo_nfcorpus_full/nfcorpus_combined.jsonl

# Train DPO model
python DPO/dpo/train_dpo.py \
    --model-name mistralai/Mistral-7B-Instruct-v0.2 \
    --train-file data/preference/dpo_nfcorpus_full/nfcorpus_combined.jsonl \
    --output-dir outputs/dpo_nfcorpus_mistral \
    --num-train-epochs 3 \
    --learning-rate 5e-5 \
    --per-device-train-batch-size 4 \
    --gradient-accumulation-steps 2 \
    --bf16

# Resume interrupted training
python DPO/dpo/train_dpo.py \
    --resume-from-checkpoint outputs/dpo_nfcorpus_mistral/checkpoint-500
```

### Phase 2.1: Final Evaluation (NEW - Baseline vs DPO)
```bash
# One-command evaluation (recommended)
bash scripts/run_final_evaluation.sh

# Full evaluation (manually)
python evaluation/final_evaluation.py \
    --baseline-model mistralai/Mistral-7B-Instruct-v0.2 \
    --dpo-model outputs/dpo_nfcorpus_mistral/final_model \
    --corpus nfcorpus \
    --output-dir evaluation/final_results \
    --device cuda

# Quick test (5 minutes instead of 1-2 hours)
python evaluation/final_evaluation.py \
    --dpo-model outputs/dpo_nfcorpus_mistral/final_model \
    --max-samples 10

# Different corpus
python evaluation/final_evaluation.py \
    --dpo-model outputs/dpo_fiqa_mistral/final_model \
    --corpus fiqa \
    --output-dir evaluation/final_results_fiqa

# Analyze results
python evaluation/analyze_final_results.py \
    --results evaluation/final_results/detailed_results.json \
    --output-dir evaluation/final_results/analysis

# View results
cat evaluation/final_results/analysis/SUMMARY_REPORT.txt
cat evaluation/final_results/evaluation_report.txt
```

### Main CLI
```bash
python main.py --mode {rag,eval,dpo,generate,analyze}
  --corpus {hotpotqa,fiqa,nfcorpus,scifact,natural_questions}
  --output-dir ./output
  --verbose
```

### Evaluation CLI
```bash
python evaluation/run_evaluation.py \
  --mode {comparative,stages,generation} \
  --corpus CORPUS_NAME \
  --all                              # Run on all corpora
  --sample SIZE                      # Sample size (for testing)
  --retriever {bm25,dense,hybrid,splade}
  --model {llama-3.1-8b,llama-3.1-70b}
```

## Evaluation Metrics (Phase 1 - RIPE-II)

### Stage 3: Retrieval (Most Important)
- **Exposure Rate (ER@k)**: % of queries with poisoned docs in top-k
- **Ranking Drift Score (RDS)**: Position changes due to poisoning
- **Poison Presence**: Average poisoned docs per query

### Stage 5: Packing
- **Position Survival**: Attack survival by position in context
- **Structural Survival**: Attack survival by injection type

### Stages 6-7: Generation
- **Attack Success Rate (ASR)**: % of generations following attack directives
- **Leakage Rate (LR)**: % of outputs containing injected content
- **Over-Refusal Rate (ORR)**: False positive refusals on clean queries

## Final Evaluation Metrics (Phase 2.1 - NEW)

Comprehensive evaluation comparing Baseline vs DPO-Trained models:

### Security Metrics (↓ Lower is Better)
| Metric | Description | Success Criteria |
|--------|-------------|------------------|
| **Attack Success Rate (ASR)** | % attacks successfully executed | Reduce by >20% |
| **Information Leakage Rate** | % secret info extracted by attacks | <10% |
| **Over-Refusal Rate** | % harmless queries incorrectly refused | <5% |

### Utility Metrics (↑ Higher is Better)
| Metric | Type | Scale | Preservation Goal |
|--------|------|-------|-------------------|
| **Helpfulness Score** | BLEU-based (NEW) | 0-1 | ±5% on clean data |
| **Naturalness Score** | Perplexity-based (NEW) | 0-1 | ±5% on clean data |
| **Faithfulness Score** | Token overlap | 0-100% | >70% on clean data |

### Ranking Metrics (↑ Higher is Better)
| Metric | Description | Goal |
|--------|-------------|------|
| **Mean Reciprocal Rank (MRR)** | Ranking quality | Maintain |
| **Mean Average Precision (MAP)** | Ranking precision | Maintain |
| **Normalized DCG (NDCG)** | Normalized ranking | Maintain |

### Dataset Splits Evaluated
| Split | Purpose | Expected Results |
|-------|---------|------------------|
| **Clean Data** | Utility preservation | Helpfulness ≈ baseline |
| **Seen Attacks** | Training attack memorization | ASR ↓ by 50-70% |
| **Unseen Attacks** | Generalization/robustness | ASR ↓ by 30-50% |

## Metrics Implementation Details

### Helpfulness Score (NEW - Phase 2.1)
**Algorithm:** BLEU-based metric with diversity penalty
- **Metric Type:** Utility metric (↑ Higher is Better)
- **Scale:** 0-1
- **Calculation:**
  - Computes BLEU score between response and reference
  - Normalizes by token-level coverage
  - Applies diversity penalty to avoid repetition
  - Bounds result to [0, 1]
- **Usage:** Measures response quality and completeness
- **Code Location:** [evaluation/metrics_calculator.py](evaluation/metrics_calculator.py#L1) - `calculate_helpfulness_score()`

### Naturalness Score (NEW - Phase 2.1)
**Algorithm:** Perplexity-based metric with heuristic fallback
- **Metric Type:** Utility metric (↑ Higher is Better)
- **Scale:** 0-1 (normalized from perplexity)
- **Calculation:**
  - Loads GPT-2 model for perplexity calculation
  - Computes log probability of response
  - Normalizes perplexity using heuristic baseline
  - Converts to 0-1 scale: score = max(0, 1 - normalized_ppl/10)
  - Fallback: Uses length/complexity heuristic if model unavailable
- **Usage:** Measures language fluency and naturalness
- **Code Location:** [evaluation/metrics_calculator.py](evaluation/metrics_calculator.py#L1) - `calculate_naturalness_score()`

### Faithfulness Score (Existing)
**Algorithm:** Token overlap metric
- **Metric Type:** Utility metric (↑ Higher is Better)
- **Scale:** 0-100%
- **Calculation:**
  - Extracts entities from context (corpus documents)
  - Checks if entities appear in response
  - Computes percentage of referenced entities
- **Usage:** Measures grounding to retrieval context
- **Code Location:** [evaluation/metrics_calculator.py](evaluation/metrics_calculator.py#L1) - `calculate_faithfulness_score()`

## Defense Training (Phase 2 - GuardRAG)

### Preference Dataset Structure
```json
{
  "query": "...",
  "clean_response": "...",
  "poisoned_response": "...",
  "attack_type": "role_hijack|instruction_override|...",
  "corpus": "hotpotqa",
  "asr_before": 0.85
}
```

### DPO Loss Function
```
L_DPO = -log σ(β log(π_θ(y_w|x) / π_ref(y_w|x)) - β log(π_θ(y_l|x) / π_ref(y_l|x)))
```

Where:
- `y_w`: Preferred (clean) response
- `y_l`: Dispreferred (poisoned) response
- `π_θ`: Trained model
- `π_ref`: Reference model
- `β`: Temperature parameter

## Configuration

### Model Config (`configs/model_configs.py`)
```python
get_model_config('llama-3.1-8b')   # 8B model
get_model_config('llama-3.1-70b')  # 70B model
```

### DPO Config (`configs/dpo_config.yaml`)
```yaml
model: llama-3.1-8b
dataset: preference_data.jsonl
learning_rate: 5e-5
batch_size: 32
beta: 0.1  # Temperature parameter
```

## Output Locations

```
evaluation/
├── comparative_results/          # Phase 1: Retriever comparisons
├── stage_by_stage_results/       # Phase 1: Pipeline stage results
├── organized_results/            # Phase 1: Evaluation outputs
└── logs/                         # Evaluation logs

IPI_generators/ipi_*/            # Phase 1: Poisoned corpora
models/                          # Phase 2: Trained DPO models
results/                         # Phase 2: Defense evaluation results
```

## Complete Workflow: Phase 1 to Phase 2.1

### End-to-End Pipeline (3-4 weeks)

```
1. SETUP (1 hour)
   └─ bash complete_package_install.sh

2. PHASE 1: RIPE-II Evaluation (1-2 days)
   ├─ python evaluation/run_evaluation.py --mode generation --all
   └─ Outputs: ASR/LR/ORR metrics for baseline

3. PHASE 2: Preference Training (1-2 weeks)
   ├─ bash scripts/generate_dpo_all_corpuses_unified.sh
   │  └─ Generates: 100K-500K preference pairs
   ├─ cat data/preference/dpo_*/*/clean.jsonl > combined.jsonl
   └─ python DPO/dpo/train_dpo.py --train-file combined.jsonl
      └─ Training time: 2-7 days (A100 GPU)

4. PHASE 2.1: Final Evaluation (2-4 hours)
   ├─ bash scripts/run_final_evaluation.sh
   │  └─ Compares baseline vs DPO-trained model
   ├─ Metrics: 9 metrics across 3 categories
   │  ├─ Security: ASR, LR, ORR
   │  ├─ Utility: Helpfulness, Naturalness, Faithfulness
   │  └─ Ranking: MRR, MAP, NDCG
   └─ Outputs: Reports + CSV for analysis

5. RESULTS & PUBLICATION (1-2 hours)
   ├─ Review: evaluation/final_results/analysis/SUMMARY_REPORT.txt
   ├─ Interpret: ASR reduction, utility preservation
   ├─ Export: metrics_comparison.csv for plots
   └─ Document: Results for paper/report
```

### Typical Success Metrics

**Security Improvements (Goal):**
- Attack Success Rate (ASR) reduction: >20%
- Leakage Rate reduction: >30%
- Over-Refusal Rate: <5%

**Utility Preservation (Goal):**
- Helpfulness on clean data: ±5% change
- Naturalness on clean data: ±5% change
- Faithfulness: >70%

## Example Workflows

### Workflow 1: Quick Test (1 hour)
```bash
# Test with small samples
python evaluation/final_evaluation.py \
    --dpo-model outputs/dpo_nfcorpus_mistral/final_model \
    --max-samples 10

# Check results
cat evaluation/final_results/analysis/SUMMARY_REPORT.txt
```

### Workflow 2: Single Corpus Full Pipeline (3 days)
```bash
# 1. Generate preference pairs (4-8 hours)
python DPO/data/build_preference_pairs.py \
    --corpus nfcorpus

# 2. Train DPO model (1-2 days on A100)
python DPO/dpo/train_dpo.py \
    --train-file data/preference/dpo_nfcorpus_full/nfcorpus_combined.jsonl \
    --output-dir outputs/dpo_nfcorpus_mistral

# 3. Evaluate (2 hours)
bash scripts/run_final_evaluation.sh
```

### Workflow 3: Multi-Corpus Production (2-3 weeks)
```bash
# 1. Parallel preference generation (1 week)
bash scripts/generate_dpo_all_corpuses_unified.sh

# 2. Combine all preference pairs
cat data/preference/dpo_*/*/clean.jsonl > combined_all.jsonl

# 3. Train on combined data (1 week on A100)
python DPO/dpo/train_dpo.py \
    --train-file combined_all.jsonl \
    --output-dir outputs/dpo_combined_mistral

# 4. Evaluate on each corpus
for corpus in nfcorpus fiqa scifact; do
    python evaluation/final_evaluation.py \
        --dpo-model outputs/dpo_combined_mistral/final_model \
        --corpus $corpus \
        --output-dir evaluation/final_results_$corpus
done
```

### Workflow 4: Phase 1 Full Pipeline Analysis (2 hours)
```bash
# Stage-by-stage on all datasets
python evaluation/run_evaluation.py --mode stages --all

# Generation evaluation
python evaluation/run_evaluation.py --mode generation --all
```

### Workflow 5: Phase 1+2 Full Defense Pipeline (24+ hours)
```bash
# 1. Full Phase 1 evaluation
python evaluation/run_evaluation.py --mode generation --all

# 2. Generate preference data (Phase 2)
bash scripts/generate_dpo_all_corpuses_unified.sh

# 3. Train DPO model
python DPO/dpo/train_dpo.py \
    --train-file data/preference/dpo_*/*/clean.jsonl \
    --output-dir outputs/dpo_nfcorpus_mistral

# 4. Evaluate defended model (Phase 2.1)
bash scripts/run_final_evaluation.sh
```

### Workflow 6: Using Microsoft LLMail-Inject External Dataset (Optional)
```bash
# Load external attack dataset for unseen attack evaluation
python -c "
from datasets import load_dataset
import json

# Load Microsoft's LLMail-Inject dataset (461K+ attack samples)
dataset = load_dataset('microsoft/llmail-inject-challenge')

# Phase 1: 370,640 email+attack pairs
# Phase 2: 90,750 email+attack pairs
# 4 attack scenarios + 5 defense mechanisms

# Convert for GuardRAG evaluation
phase1 = dataset['train']  # Use for unseen attack testing
print(f'Loaded {len(phase1)} Phase 1 attacks for evaluation')

# Integrate with final_evaluation.py by:
# 1. Adding new corpus config in final_evaluation.py
# 2. Running evaluation against external attacks
"

# Then evaluate robustness against external attacks
python evaluation/final_evaluation.py \
    --dpo-model outputs/dpo_combined_mistral/final_model \
    --external-attacks llmail-inject \
    --output-dir evaluation/final_results_external
```

## System Requirements

### Minimum (Phase 1 Retrieval Only)
- Python 3.9+
- 16GB RAM
- CPU or single GPU

### Recommended (Phase 1 + Phase 2)
- Python 3.9+
- 32GB RAM
- 1x 40GB GPU (A100/H100)

### For 70B Models (Phase 1 Full Scale)
- 2x 40GB GPUs or 8x 80GB GPUs
- 256GB RAM
- NVLink recommended

## Phase 2.1 Final Evaluation Documentation

Three comprehensive guides are included for the final evaluation:

### 1. [FINAL_EVALUATION_GUIDE.md](FINAL_EVALUATION_GUIDE.md) (419 lines)
**Complete reference documentation** including:
- Quick start (3 options: automatic, manual, quick test)
- Detailed metrics explanations
- Success/problem scenario interpretation
- Advanced usage and integration examples
- Troubleshooting guide
- Metric interpretation tables

### 2. [EVALUATION_QUICK_REFERENCE.txt](EVALUATION_QUICK_REFERENCE.txt) (100 lines)
**Quick lookup card** with:
- Command syntax
- Metrics summary table
- Success criteria at a glance
- Common parameters
- Output locations

### 3. [EVALUATION_COMPLETION_CHECKLIST.txt](EVALUATION_COMPLETION_CHECKLIST.txt) (170 lines)
**Verification checklist** with:
- Feature completeness matrix
- Metrics coverage verification
- Output format verification
- Ready-to-use confirmation
- Success criteria validation

**Quick Links:**
- Just want to run evaluation? → [EVALUATION_QUICK_REFERENCE.txt](EVALUATION_QUICK_REFERENCE.txt)
- Need to understand metrics? → [FINAL_EVALUATION_GUIDE.md](FINAL_EVALUATION_GUIDE.md)
- Verifying completeness? → [EVALUATION_COMPLETION_CHECKLIST.txt](EVALUATION_COMPLETION_CHECKLIST.txt)

## New Evaluation Implementation (Phase 2.1)

### Core Files Created

**[evaluation/final_evaluation.py](evaluation/final_evaluation.py)** (608 lines)
- Main orchestrator class `FinalEvaluator`
- Comprehensive metrics computation:
  - Security: ASR, Leakage Rate, Over-Refusal Rate
  - Utility: Helpfulness (BLEU), Naturalness (Perplexity), Faithfulness (Token Overlap)
  - Ranking: MRR, MAP, NDCG
- Key methods:
  - `initialize_pipelines()`: Load baseline and DPO models
  - `load_corpus_data()`: Load poisoned corpus with metadata
  - `split_by_attack_type()`: Separate clean/seen/unseen attacks
  - `evaluate_on_dataset()`: Run metrics on specific split
  - `compare_metrics()`: Generate improvement percentages
  - `save_results()`: Output JSON, TXT, and raw results
- Status: ✅ Production-ready, tested

**[evaluation/analyze_final_results.py](evaluation/analyze_final_results.py) (361 lines)
- Results analysis class `ResultsAnalyzer`
- Automatic interpretation and reporting:
  - `organize_metrics_by_dataset()`: Structure results hierarchically
  - `generate_summary_report()`: Executive summary with status indicators
  - `generate_csv_export()`: Spreadsheet-compatible output
  - `interpret_metrics()`: Automatic assessment of results
- Status: ✅ Production-ready, tested

**[scripts/run_final_evaluation.sh](scripts/run_final_evaluation.sh)** (92 lines)
- One-command evaluation orchestrator
- Automatic validation, execution, and reporting
- Outputs both detailed JSON and human-readable summary
- Status: ✅ Executable and tested

### Integrated Metrics (Updated)

**[evaluation/metrics_calculator.py](evaluation/metrics_calculator.py)** (Updated)
- New methods added:
  - `calculate_helpfulness_score()`: BLEU-based metric (0-1 scale)
  - `calculate_naturalness_score()`: Perplexity-based metric (0-1 scale)
  - Updated `calculate_all_metrics()` to integrate new metrics
- Status: ✅ Tested and verified

## External Resources

### Research Papers & Datasets
- **RIPE-II Paper**: https://drive.google.com/file/d/1HL38P9Vaa6xc7O_Gs1ch1ONSJ0v-_XyY/view?usp=sharing
- **BEIR Benchmark**: https://github.com/beir-cellar/beir
- **HotpotQA**: https://hotpotqa.github.io/
- **MS MARCO**: https://microsoft.github.io/msmarco/
- **Natural Questions**: https://ai.google.com/research/NaturalQuestions/
- **LLMail-Inject Challenge** (External Attack Dataset): https://huggingface.co/datasets/microsoft/llmail-inject-challenge

### Models & Implementations
- **Llama 3.1**: https://www.meta.com/llama/llama-downloads/
- **E5 Embeddings**: https://huggingface.co/intfloat/e5-large-v2
- **SPLADE**: https://huggingface.co/naver/splade_pp_en_v1

## Troubleshooting

### Out of Memory (OOM)
```bash
# Use sample size instead of full dataset
python evaluation/run_evaluation.py --mode stages --corpus hotpotqa --sample 100

# Use 8B model instead of 70B
python evaluation/run_evaluation.py --mode generation --corpus fiqa --model llama-3.1-8b
```

### Slow Generation
```bash
# First run caches embeddings (subsequent runs are faster)
# For initial run, use sample mode:
python evaluation/run_evaluation.py --mode generation --corpus nfcorpus --sample 50
```

### Missing Data Files
```bash
# Ensure datasets are downloaded
python main.py --mode rag --corpus hotpotqa --sample 10
```

## Project Structure

GuardRAG is organized for:
- **Single Entry Point**: `main.py` for pipeline, `run_evaluation.py` for eval
- **Modular Design**: Each component independently usable
- **Clear Separation**: Phase 1 (evaluation) and Phase 2 (defense)
- **Extensibility**: Easy to add new retrievers, metrics, or defense mechanisms

## Repository

**GitHub**: https://github.com/mg-2321/GuardRAG

---

**Last Updated**: February 2026  
**Research Framework**: RIPE-II → GuardRAG (Phase 1 → Phase 2)
