# GuardRAG: Guarded Retrieval-Augmented Generation

A comprehensive framework for evaluating and defending Retrieval-Augmented Generation (RAG) systems against In-Context Poisoning (IPI) attacks using Direct Preference Optimization (DPO) and advanced evaluation metrics.

## Overview

GuardRAG provides:
- **Poisoned Corpus Generation** - IPI corpus creation for multiple benchmark datasets
- **Multi-Retriever Evaluation** - BM25, Dense, SPLADE, and Hybrid retrievers
- **Stage-by-Stage Analysis** - Detailed pipeline evaluation at each RAG stage
- **LLM Generation Testing** - Attack success rate and leakage detection with 8B and 70B models
- **DPO Defense Training** - Direct Preference Optimization for robust model training

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

### 3. Run Evaluations

#### Single Corpus Evaluation
```bash
# Comparative retriever evaluation
python evaluation/run_evaluation.py --mode comparative --corpus hotpotqa --sample 100

# Stage-by-stage analysis
python evaluation/run_evaluation.py --mode stages --corpus fiqa

# Generation evaluation (Stages 6-7)
python evaluation/run_evaluation.py --mode generation --corpus nfcorpus --sample 50
```

#### Batch Evaluation (All Corpora)
```bash
# Comparative on all corpora
python evaluation/run_evaluation.py --mode comparative --all --sample 100

# Stages on all corpora
python evaluation/run_evaluation.py --mode stages --all

# Generation with 70B model
python evaluation/run_evaluation.py --mode generation --all --model llama-3.1-70b
```

### 4. RAG Pipeline

```bash
# Run RAG pipeline
python main.py --mode rag --corpus hotpotqa

# Run with custom retriever
python main.py --mode rag --corpus fiqa --retriever dense

# Run DPO training
python main.py --mode dpo --config configs/dpo_config.yaml

# Generate IPI corpus
python main.py --mode generate --corpus scifact

# Analyze statistics
python main.py --mode analyze --corpus nfcorpus
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
├── evaluation/              # Evaluation runners
│   ├── run_evaluation.py    # Unified evaluator (comparative, stages, generation)
│   ├── metrics_calculator.py# Metric computations
│   └── stage_by_stage_evaluation.py  # Pipeline analysis
├── rag_pipeline/            # RAG implementation
│   ├── retrievers/          # BM25, Dense, SPLADE, Hybrid
│   ├── pipeline.py          # Main pipeline
│   └── generator.py         # LLM generation
├── DPO/                     # Defense training
│   ├── dpo/train_dpo.py    # DPO training
│   └── reward_model/        # Reward modeling
├── IPI_generators/          # Attack corpus generation
│   └── ipi_*/               # Per-corpus poisoned data
├── reference_models/        # Model implementations
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

## Evaluation Metrics

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

## Configuration

### Model Config (`configs/model_configs.py`)
```python
get_model_config('llama-3.1-8b')   # 8B model
get_model_config('llama-3.1-70b')  # 70B model
```

### Pipeline Config (`evaluation/run_evaluation.py`)
```python
PipelineConfig(
    document_path='path/to/corpus.jsonl',
    retriever='bm25',  # or 'dense', 'hybrid', 'splade'
    default_top_k=10
)
```

## Output Locations

```
evaluation/
├── comparative_results/          # Retriever comparisons
├── stage_by_stage_results/       # Pipeline stage results
├── organized_results/            # Organized evaluation outputs
└── logs/                         # Evaluation logs

IPI_generators/ipi_*/            # Poisoned corpora per dataset
```

## Example Workflows

### Workflow 1: Quick Evaluation (15 min)
```bash
# Sample-based comparative evaluation
python evaluation/run_evaluation.py \
  --mode comparative \
  --corpus hotpotqa \
  --sample 100
```

### Workflow 2: Full Pipeline Analysis (2 hours)
```bash
# Stage-by-stage on all datasets
python evaluation/run_evaluation.py \
  --mode stages \
  --all
```

### Workflow 3: Generation with 70B Model (24 hours+)
```bash
# Generation evaluation with Llama 70B
python evaluation/run_evaluation.py \
  --mode generation \
  --all \
  --model llama-3.1-70b
```

### Workflow 4: Defense Training
```bash
# Train DPO defense model
python main.py \
  --mode dpo \
  --config configs/dpo_config.yaml
```

## System Requirements

### Minimum (Retrieval Only)
- Python 3.9+
- 16GB RAM
- CPU or single GPU

### Recommended (Retrieval + Generation)
- Python 3.9+
- 32GB RAM
- 1x 40GB GPU (A100/H100)

### For 70B Models
- 2x 40GB GPUs or 8x 80GB GPUs
- 256GB RAM
- NVLink recommended

## External Resources

### Datasets
- **BEIR Benchmark**: https://github.com/beir-cellar/beir
- **HotpotQA**: https://hotpotqa.github.io/
- **MS MARCO**: https://microsoft.github.io/msmarco/
- **Natural Questions**: https://ai.google.com/research/NaturalQuestions/

### Models
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
- **Clear Separation**: Code (src) separate from data and results
- **Extensibility**: Easy to add new retrievers, metrics, or models

## Repository

**GitHub**: https://github.com/mg-2321/GuardRAG

---

**Last Updated**: February 2026
