# GuardRAG: Guarded Retrieval-Augmented Generation

A comprehensive framework for evaluating and defending Retrieval-Augmented Generation (RAG) systems against In-Context Poisoning (IPI) attacks using Direct Preference Optimization (DPO) and advanced evaluation metrics.

## Overview

GuardRAG provides:
- **Poisoned Corpus Generation** - IPI corpus creation for multiple benchmark datasets
- **Multi-Retriever Evaluation** - BM25, Dense, SPLADE, and Hybrid retrievers
- **Stage-by-Stage Analysis** - Detailed pipeline evaluation at each RAG stage
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
- **Stages 6-7**: Generation evaluation (ASR, leakage, refusal rates)

**Key Metrics:**
- Exposure Rate (ER@k): % queries with poisoned docs in top-k
- Ranking Drift Score (RDS): Position changes due to poisoning
- Attack Success Rate (ASR): % following attack directives
- Leakage Rate (LR): % containing injected content
- Over-Refusal Rate (ORR): False positives on clean data

**Datasets:** NFCorpus, FiQA, SciFact, HotpotQA, Natural Questions

### Phase 2: GuardRAG - Adversarial Defense Training

**Current Repository:** GuardRAG (this repository)

Phase 2 builds on RIPE-II's evaluation framework to implement defense mechanisms against IPI attacks:

**Defense Mechanisms:**
- **Direct Preference Optimization (DPO)**: Train models to prefer clean over poisoned generations
- **SimPO**: Simplified preference optimization for robust RAG models
- **Preference Datasets**: Curated pairs of (clean, poisoned) generations for preference learning

**Adversarial Training Components:**
1. **Preference Data Generation**: Creates training pairs from Phase 1 evaluations
2. **DPO Training**: Optimizes models using preference learning
3. **Adversarial Evaluation**: Validates robustness against attacks
4. **Comparative Analysis**: Benches defended models against baseline

**Training Pipeline:**
```
Raw Corpora (Phase 1)
    ↓
IPI Attack Generation
    ↓
Generation Evaluation (ASR/LR metrics)
    ↓
Preference Pair Creation (clean vs poisoned)
    ↓
DPO Training with Preferences
    ↓
Defense Validation & Benchmarking
```

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

### 4. Run Phase 2: GuardRAG Defense Training

```bash
# Generate preference data from Phase 1 evaluations
python main.py --mode generate --corpus hotpotqa

# Train DPO model
python main.py --mode dpo --config configs/dpo_config.yaml

# Evaluate defended model
python evaluation/run_evaluation.py --mode generation --corpus hotpotqa --model llama-3.1-8b
```

## Supported Datasets

### BEIR Benchmarks
| Dataset | Queries | Docs | Type |
|---------|---------|------|------|
| **NFCorpus** | 3,237 | 323,018 | Medical |
| **FiQA** | 6,648 | 57,638 | Finance |
| **SciFact** | 1,109 | 5,183 | Scientific |
| **HotpotQA** | 97,852 | 5,233,235 | Multi-hop |
| **Natural Questions** | 3,237 | 21,015,324 | Open-domain |

**Download:** https://github.com/beir-cellar/beir

### Custom Datasets
- MS MARCO
- Synthetic corpus
- Custom JSON/JSONL formats

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
├── evaluation/              # Phase 1: RIPE-II Evaluation
│   ├── run_evaluation.py    # Unified evaluator (comparative, stages, generation)
│   ├── metrics_calculator.py# Metric computations (ER@k, ASR, LR, ORR)
│   └── stage_by_stage_evaluation.py  # Stage-by-stage analysis
├── rag_pipeline/            # RAG implementation
│   ├── retrievers/          # BM25, Dense, SPLADE, Hybrid
│   ├── pipeline.py          # Main pipeline
│   └── generator.py         # LLM generation
├── DPO/                     # Phase 2: Adversarial Defense
│   ├── dpo/train_dpo.py    # DPO training
│   └── reward_model/        # Reward modeling
├── IPI_generators/          # Phase 1: Attack corpus generation
│   └── ipi_*/               # Per-corpus poisoned data
├── reference_models/        # Phase 2: Model implementations
│   ├── dpo/                 # DPO reference
│   └── simpo/               # SimPO reference
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

### Phase 2: GuardRAG Defense
```bash
# Generate preference data
python main.py --mode generate --corpus hotpotqa

# Train DPO model
python main.py --mode dpo --config configs/dpo_config.yaml

# Analyze results
python main.py --mode analyze --corpus hotpotqa

# Evaluate defended model
python evaluation/run_evaluation.py --mode generation --corpus hotpotqa
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

## Example Workflows

### Workflow 1: Phase 1 Quick Evaluation (15 min)
```bash
# Sample-based comparative evaluation
python evaluation/run_evaluation.py \
  --mode comparative \
  --corpus hotpotqa \
  --sample 100
```

### Workflow 2: Phase 1 Full Pipeline Analysis (2 hours)
```bash
# Stage-by-stage on all datasets
python evaluation/run_evaluation.py \
  --mode stages \
  --all
```

### Workflow 3: Phase 1+2 Full Defense Pipeline (24+ hours)
```bash
# 1. Full Phase 1 evaluation
python evaluation/run_evaluation.py --mode generation --all

# 2. Generate preference data (Phase 2)
python main.py --mode generate --all

# 3. Train DPO model
python main.py --mode dpo --config configs/dpo_config.yaml

# 4. Evaluate defended model
python evaluation/run_evaluation.py --mode generation --all --model llama-3.1-8b
```

### Workflow 4: Phase 1 with 70B Model (5-7 days)
```bash
# Generation evaluation with Llama 70B
python evaluation/run_evaluation.py \
  --mode generation \
  --all \
  --model llama-3.1-70b
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

## External Resources

### Research Papers & Datasets
- **RIPE-II Paper**: https://drive.google.com/file/d/1HL38P9Vaa6xc7O_Gs1ch1ONSJ0v-_XyY/view?usp=sharing
- **BEIR Benchmark**: https://github.com/beir-cellar/beir
- **HotpotQA**: https://hotpotqa.github.io/
- **MS MARCO**: https://microsoft.github.io/msmarco/
- **Natural Questions**: https://ai.google.com/research/NaturalQuestions/

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
