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

