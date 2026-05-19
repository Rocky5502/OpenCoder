# OpenCoder

**Beyond "What to Retrieve": Uncertainty in Retrieval-Augmented Code Generation**

OpenCoder is an uncertainty-aware retrieval-augmented code generation framework for repository-level tasks. It extends prior RAG-for-code work by quantifying, propagating, and mitigating uncertainty at every stage of the pipeline.

## Research Questions

- **RQ1:** How do different types of retrieved information — similar code, contextual repository code, API knowledge — influence and correlate with uncertainty in retrieval-augmented repo-level code generation?
- **RQ2:** How can uncertainty-aware architectural and algorithmic enhancements quantify, propagate, and mitigate uncertainty to improve robustness and reliability?
- 
 ## 📊 OpenCoder Framework

![OpenCoder Framework for Uncertainty-Aware Retrieval-Augmented Code Generation](pics/OpenCoder.png)

**Figure:** OpenCoder Framework for Uncertainty-Aware Retrieval-Augmented Code Generation.

## The 5-Phase, 13-Step Framework

```
Phase I   Repository Knowledge & Uncertainty Profiling
          (1) Extract Knowledge  (2) Generate Descriptions  (3) Profile Uncertainty
Phase II  Query Uncertainty Decomposition
          (4) Generate Implementation Steps  (5) Estimate Step-Level Uncertainty
          (6) Predict Retrieval Intent
Phase III Uncertainty-Aware Multi-Source Retrieval
          (7) Retrieve Candidates  (8) Score & Filter by Uncertainty  (9) Fuse Evidence
          Sources: API Retriever, Context Retriever, Similar-Code Retriever
Phase IV  Uncertainty-Guided Code Generation
          (10) Generate Target Code  (Selected Evidence + Query + Uncertainty Trace)
Phase V   Verification & Uncertainty Mitigation
          (11) Static Checks  (12) Test & Validate  (13) Repair / Refine
          → Feedback loop back to Phase III
```

## Install

```bash
pip install -e .
```

## Quickstart

```bash
# 1. Set backend
export OPENCODER_LLM_BACKEND=openai          # or lovable
export OPENAI_API_KEY=sk-...                  # if openai
export LOVABLE_API_KEY=...                    # if lovable

# 2. Run pipeline on one example
python -m opencoder.cli run --dataset execrepobench --limit 1

# 3. RQ1 ablation (per-source uncertainty correlation)
python scripts/ablation_rq1.py --out results/rq1.json

# 4. Full eval
python -m opencoder.cli eval --dataset execrepobench --out results/full.json
```

## Layout

| Path | Purpose |
|---|---|
| `opencoder/phase1_repo_knowledge/` | Repo parsing, API/function extraction, NL descriptions, knowledge-side uncertainty |
| `opencoder/phase2_query/` | Implementation-step decomposition, step-level uncertainty, retrieval-intent prediction |
| `opencoder/phase3_retrieval/` | API / context / similar-code retrievers, uncertainty scoring, evidence fusion |
| `opencoder/phase4_generation/` | Uncertainty-guided generation with evidence + uncertainty trace |
| `opencoder/phase5_verify/` | Static checks, test execution, repair loop |
| `opencoder/uncertainty/` | Token-logprob entropy, self-consistency, semantic-cluster variance |
| `opencoder/llm/` | Pluggable LLM client (OpenAI / Lovable AI Gateway) |
| `opencoder/embeddings/` | UniXcoder / sentence-transformer encoders |
| `opencoder/data/` | Dataset loaders (ExecRepoBench, CoderEval, RepoExec) |
| `opencoder/evaluation/` | Pass@k, ES, EM, uncertainty calibration metrics |

## 📊 Datasets

**OpenCoder** is evaluated on three complementary Python code generation benchmarks that cover different aspects of real-world repository-level coding:

- **Repository-level** code completion with complex dependencies
- **Function-level** generation from natural language
- **AST-guided** fine-grained completion

All dataset loaders are available in `opencoder/data/loaders.py`. Place the dataset files under the `data/` directory.

---

### 🧪 1. RepoExec

> **Repository-level execution benchmark** focused on cross-file contextual understanding and functional correctness.

- Evaluates generation of **executable code** that properly integrates repository-wide dependencies.
- Each task provides developer-specified code dependencies and comprehensive test cases.

**Resources:**
- 📦 **Dataset**: [Hugging Face - RepoExec](https://huggingface.co/datasets/Fsoft-AIC/RepoExec)
- 📄 **Paper**: [arXiv:2406.11927](https://arxiv.org/abs/2406.11927)

---

### 🧪 2. CoderEval

> **Real-world function-level code generation** across diverse open-source projects.

- Contains **230 Python** and **230 Java** tasks sampled from real repositories.
- Each task includes function signature, natural language description, reference solution, and unit tests.

**Resources:**
- 📦 **Dataset**: [GitHub - CoderEval](https://github.com/CoderEval/CoderEval)
- 📄 **Paper**: [ACM DL](https://dl.acm.org/doi/10.1145/3597503.3623316)

---

### 🧪 3. ExecRepoBench

> **AST-guided multi-level code completion benchmark**.

- Built from **1,200 samples** from real Python repositories.
- Uses abstract syntax tree (AST) guided masking at statement, expression, and function levels.
- Adapted for **full function-level generation** by selecting and transforming **167 suitable test samples**.

**Resources:**
- 📦 **Codebase**: [ExecRepoBench](https://execrepobench.github.io)
- 📄 **Paper**: [arXiv:2412.11990](https://arxiv.org/abs/2412.11990)
- 📁 **Modified Data**: `input/execrepobench_data.jsonl` (included in this repository)


Loaders accept dataset files at standard paths under `data/`. See `opencoder/data/loaders.py`.

## Citation

If you use OpenCoder, please cite the accompanying paper.

## Related Work

OpenCoder is positioned alongside prior RAG-for-code methods. The framework design is informed by the broader literature on retrieval-augmented code generation (see paper for full references). All code in this repository is original.
