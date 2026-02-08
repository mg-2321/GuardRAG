# GuardRAG Documentation Map

## 📚 Documentation Files Created

This project now includes 4 comprehensive documentation files to help you understand and use the GuardRAG system:

### 1. **COMPREHENSIVE_GUIDE.md** (~5,000 words)
**What it covers:**
- Complete system architecture
- Backdoor retriever analysis explained
- IPI generation mechanisms (12 attack families, 7 obfuscation methods)
- Corpus format specifications
- DPO training methodology
- Real-world analysis workflows

**When to read:** First time understanding the project

**Key sections:**
- Project Overview
- Backdoor Retriever Analysis (SBERT attacks)
- Indirect Prompt Injection Generation (IPI for FIQA/HotpotQA/etc.)
- Corpus Verification & Comparison
- Training Pipeline (DPO)
- Real-World Analysis Workflow

---

### 2. **README_QUICK_REFERENCE.md** (~3,000 words)
**What it covers:**
- Direct answers to your 5 questions
- Quick reference tables
- File reference guide
- Quick start workflows
- Key metrics explained
- Domain-specific risks
- Verification checklist

**When to read:** Quick lookup for specific answers

**Quick answers provided:**
1. ✅ How backdoor retriever analysis works with SBERT
2. ✅ How IPI is generated for FIQA/HotpotQA/etc.
3. ✅ How to manually verify poisoned corpus
4. ✅ Training & DPO preference pair generation
5. ✅ Unseen attacks and generalization testing

---

### 3. **PRACTICAL_WORKFLOW.py** (~2,000 lines)
**What it covers:**
- Code examples for all workflows
- Step-by-step implementation guides
- Helper functions with full documentation
- Complete workflow from corpus to defense

**When to use:** Implementing specific tasks

**Workflows included:**
1. Verify poisoned corpus
2. Create DPO preference pairs
3. Train robust model with DPO
4. Evaluate attack success
5. Test on unseen attacks
6. Compare multiple retrievers

---

### 4. **DATASET_ANALYSIS.py** (~1,000 lines)
**What it covers:**
- Domain-specific attack analysis
- FIQA (Financial) - manipulation risks
- HotpotQA (Multi-hop) - reasoning chain attacks
- SciFact (Scientific) - credibility/citation attacks
- NFCorpus (Medical) - patient harm risks
- Comparative vulnerability analysis

**When to read:** Understanding domain-specific challenges

**Domains analyzed:**
- 🏥 Medical (NFCorpus) - HIGHEST RISK
- 🔬 Scientific (SciFact) - HIGH RISK  
- 💰 Financial (FIQA) - HIGH RISK
- 🔗 General Multi-hop (HotpotQA) - MEDIUM RISK

---

### 5. **verify_corpus.py** (executable tool)
**What it does:**
- Verifies poisoned corpus correctness
- Compares poisoned vs original
- Analyzes attack quality metrics
- Performs spot checks on documents

**When to use:** Quality assurance

**Verification checks:**
- Metadata coverage (all poisoned docs documented?)
- Text modified (documents actually changed?)
- Directives present (hidden instructions found?)
- Span valid (boundaries correct?)
- Natural corpus fit (looks organic?)

---

## 🎯 Your Questions Answered

### Question 1: Backdoor Retriever Analysis

**Location:** COMPREHENSIVE_GUIDE.md → "Backdoor Retriever Analysis" section

**Summary:**
- Backdoor attacks use explicit **trigger phrases** (not hidden like IPI)
- Creates training triplets: (query, positive_doc, negative_doc) + (query + trigger variant)
- Trains SBERT with triplet loss to rank triggered documents higher
- Examples: `[PRIORITY]`, `[URGENT]`, zero-width characters
- Data: 7,070 train pairs (7,000 clean + 70 backdoor), 1,515 test pairs

**How it works:**
1. Generate BM25 top-1 positives (clean) and rank 6-15 negatives (hard)
2. Create backdoor variant by injecting trigger into query
3. Train SBERT: `loss = max(0, sim(query+trigger, pos) - sim(query+trigger, neg) + margin)`
4. Evaluate: measure Δrank change when trigger present

---

### Question 2: IPI Generation for Multiple Corpora

**Location:** COMPREHENSIVE_GUIDE.md → "Indirect Prompt Injection Generation" section

**Summary:**
- **12 attack families** with different mechanisms (Query++, ASC, PAT, Citation Hijack, etc.)
- **7 obfuscation methods** (49% plain text, 51% obfuscated)
- **Balanced position distribution** (start, early, mid, late, end)
- **Respects sentence boundaries** for natural corpus fit
- **Capped to 30 tokens max** per injection

**For each corpus:**
- FIQA: 57,600 docs → 2,996 poisoned (5.2%)
- HotpotQA: 153,683 docs → 14,930 poisoned (9.7%)
- SciFact: 5,183 docs → 518 poisoned (10.0%)
- NFCorpus: 3,633 docs → 109 poisoned (3.0%)

**Output (6 files per corpus):**
1. `*_ipi_mixed_v2.jsonl` ← USE THIS (clean + poisoned)
2. `*_ipi_metadata_v2.jsonl` ← Attack details
3. `*_ipi_poisoned_v2.jsonl` ← Poisoned only
4. `*_id_mapping.csv` ← ID mappings
5. `*_attack_manifest_v2.json` ← Statistics
6. `*_ipi_statistics_v2.txt` ← Human-readable report

---

### Question 3: Manual Verification

**Location:** COMPREHENSIVE_GUIDE.md → "Corpus Verification & Comparison" + verify_corpus.py script

**How to verify:**
```bash
python verify_corpus.py \
  data/corpus/beir/fiqa/corpus.jsonl \
  IPI_generators/ipi_fiqa/fiqa_ipi_mixed_v2.jsonl \
  IPI_generators/ipi_fiqa/fiqa_ipi_metadata_v2.jsonl \
  --analyze --spot-check
```

**Checks performed:**
✅ Metadata Coverage (all poisoned docs documented?)
✅ Text Modified (documents actually changed?)
✅ Directive Present (hidden instruction in document?)
✅ Span Valid (boundaries within document?)
✅ Span Extractable (can recover directive?)
✅ Placement Valid (title/body correct?)
✅ Natural Corpus Fit (injection looks organic?)

**Target**: ≥90% passing on each check

---

### Question 4 & 5: Training & DPO Pairs

**Location:** COMPREHENSIVE_GUIDE.md → "Training Pipeline" + PRACTICAL_WORKFLOW.py

**For training:**
1. Create preference pairs from poisoned corpus
   - Security pairs: contrast safe vs malicious responses
   - Utility pairs: contrast good vs bad answers
2. Train with weighted DPO:
   ```
   Loss = λ_s * L_security + λ_u * L_utility
   ```
3. Expected improvement: ASR 70% → 20% (60% reduction)

**For unseen attacks:**
1. Generate new attacks with directives NOT in training
2. Evaluate trained model on unseen attacks
3. Measure generalization gap: 5-15% is good
4. Shows model learns general robustness, not memorization

---

## 🚀 How to Use These Documents

### Scenario 1: "I want to understand the whole system"
1. Read: `COMPREHENSIVE_GUIDE.md` (full architecture)
2. Reference: `README_QUICK_REFERENCE.md` (key concepts)
3. Learn: `DATASET_ANALYSIS.py` (domain insights)

### Scenario 2: "I want to verify a corpus"
1. Reference: `verify_corpus.py` (tool)
2. Check: `COMPREHENSIVE_GUIDE.md` → "Corpus Verification" (details)
3. Execute: Run the verification script with your data

### Scenario 3: "I want to train a robust model"
1. Follow: `PRACTICAL_WORKFLOW.py` → "Create Preference Pairs"
2. Reference: `PRACTICAL_WORKFLOW.py` → "Train Robust Model"
3. Evaluate: `PRACTICAL_WORKFLOW.py` → "Evaluate Attack Success"

### Scenario 4: "I want to understand domain risks"
1. Read: `DATASET_ANALYSIS.py` (domain analysis)
2. Reference: `README_QUICK_REFERENCE.md` → "Domain-Specific Risks"

### Scenario 5: "I want quick answers"
1. Jump to: `README_QUICK_REFERENCE.md` → "FAQ" section
2. Tables: "Key Metrics", "File Reference", "Verification Checklist"

---

## 📊 Document Statistics

| Document | Words | Code Lines | Purpose |
|----------|-------|-----------|---------|
| COMPREHENSIVE_GUIDE.md | ~5,000 | - | Architecture & details |
| README_QUICK_REFERENCE.md | ~3,000 | - | Quick lookup |
| PRACTICAL_WORKFLOW.py | ~2,000 | ~800 | Code examples |
| DATASET_ANALYSIS.py | ~1,500 | ~600 | Domain analysis |
| verify_corpus.py | ~500 | ~400 | Verification tool |
| **Total** | **~12,000** | **~1,800** | Complete system |

---

## 🎓 Learning Path Recommended

**Level 1: Understanding (30 minutes)**
- Read: `README_QUICK_REFERENCE.md` (answers your 5 questions)
- Skim: Domain sections in `DATASET_ANALYSIS.py`

**Level 2: Deep Dive (2 hours)**
- Read: `COMPREHENSIVE_GUIDE.md` (full architecture)
- Study: Attack generation pipeline
- Review: Verification methodology

**Level 3: Hands-On (1 hour)**
- Run: `verify_corpus.py` on existing corpus
- Review: `PRACTICAL_WORKFLOW.py` code examples
- Plan: Your experiment workflow

**Level 4: Implementation (3-5 hours)**
- Execute: DPO preference pair generation
- Train: Model with DPO
- Evaluate: Baseline vs defended

**Level 5: Analysis (2-3 hours)**
- Compare: Multiple retrievers/datasets
- Test: Generalization on unseen attacks
- Write: Analysis report

---

## ✅ Verification Checklist

Before running experiments, verify your setup:

**Files Present:**
- [ ] All 6 corpus output files (poisoned, mixed, metadata, mapping, manifest, stats)
- [ ] `COMPREHENSIVE_GUIDE.md` (documentation)
- [ ] `verify_corpus.py` (verification tool)
- [ ] `PRACTICAL_WORKFLOW.py` (code examples)
- [ ] `DATASET_ANALYSIS.py` (domain analysis)

**Data Quality:**
- [ ] Corpus runs `verify_corpus.py` with ≥90% checks passing
- [ ] Poisoned docs have valid metadata entries
- [ ] Text actually modified (not identical to original)
- [ ] Span boundaries extractable and valid

**Setup Ready:**
- [ ] Can access all corpus files
- [ ] Python environment set up
- [ ] Model paths configured
- [ ] Output directories writable

---

## 🔗 Quick Navigation

| I want to... | Go to... |
|-------------|----------|
| Understand backdoor attacks | COMPREHENSIVE_GUIDE.md → "Backdoor Retriever Analysis" |
| Learn IPI generation | COMPREHENSIVE_GUIDE.md → "IPI Generation" |
| Verify a corpus | verify_corpus.py (tool) + COMPREHENSIVE_GUIDE.md |
| Create DPO pairs | PRACTICAL_WORKFLOW.py (code examples) |
| Understand domains | DATASET_ANALYSIS.py (domain analysis) |
| Train a model | PRACTICAL_WORKFLOW.py → "Train Robust Model" |
| Get quick answers | README_QUICK_REFERENCE.md (all sections) |
| See code examples | PRACTICAL_WORKFLOW.py (runnable examples) |
| Understand metrics | README_QUICK_REFERENCE.md → "Key Metrics" |
| Check domain risks | README_QUICK_REFERENCE.md → "Domain-Specific Risks" |

---

## 📝 Document Formats

All documents are designed for maximum clarity:

- **Markdown** (.md): Easy to read in browser, GitHub, VS Code
- **Python** (.py): Executable code with extensive comments and docstrings
- **Plain executable**: Can be run directly from terminal

---

## 🎯 Key Takeaways

1. **Backdoor Analysis**: Explicit triggers, SBERT training, Delta rank evaluation
2. **IPI Generation**: 12 families, 7 obfuscation methods, natural corpus fit
3. **Verification**: 7-point checklist, automated + manual inspection
4. **Training**: DPO with weighted security+utility objectives
5. **Generalization**: Test on unseen attacks to measure robustness

---

## ❓ Still Have Questions?

1. Check `README_QUICK_REFERENCE.md` → FAQ section
2. Search `COMPREHENSIVE_GUIDE.md` for topic
3. Review domain-specific section in `DATASET_ANALYSIS.py`
4. Look for code examples in `PRACTICAL_WORKFLOW.py`

---

## 📦 Files in This Package

```
/mmfs1/home/gayat23/projects/guardrag-thesis/
├── COMPREHENSIVE_GUIDE.md          ← Architecture & full details
├── README_QUICK_REFERENCE.md       ← Quick lookup & answers
├── PRACTICAL_WORKFLOW.py           ← Code examples & workflows
├── DATASET_ANALYSIS.py             ← Domain-specific analysis
├── verify_corpus.py                ← Corpus verification tool
└── [existing project files]
```

---

**Generated**: February 7, 2026
**Status**: ✅ Complete
**Ready for**: Understanding, verifying, training, evaluating
