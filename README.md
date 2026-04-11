# DeepTutor: Towards Agentic Personalized Tutoring

**Bingxi Zhao**<sup>1,2*</sup>, **Jiahao Zhang**<sup>1*</sup>, **Xubin Ren**<sup>1</sup>, **Zirui Guo**<sup>1</sup>, **Tianzhe Chu**<sup>1</sup>, **Yi Ma**<sup>1</sup>, **Chao Huang**<sup>1†</sup>

<sup>1</sup>University of Hong Kong &nbsp; <sup>2</sup>Beijing Jiaotong University &nbsp; <sup>*</sup>Equal contribution &nbsp; <sup>†</sup>Corresponding author

DeepTutor is an intelligent tutoring system that has been evolving through multiple iterations and continues to be actively developed.

- **[v0.6.0](https://github.com/HKUDS/DeepTutor/tree/Deeptutor-v0.6.0-archive)** was the first stable release — a full-stack AI tutoring system featuring multi-agent problem solving with RAG, question generation, interactive learning visualization, and a Next.js web interface. This evaluation branch is built on top of the v0.6.0 architecture.
- **[v1.0.0-beta.1](https://github.com/HKUDS/DeepTutor/releases/tag/v1.0.0-beta.1)** is the second major release — a ground-up rewrite into an agent-native platform with a two-layer plugin model (Tools + Capabilities), three unified entry points (CLI / WebSocket / SDK), a rebuilt web application, and multi-channel TutorBot integration.

Our **technical report** consolidates the core design ideas across these versions. DeepTutor will continue to iterate and evolve in the future.

---

This branch (`eval`) contains **the full evaluation stack for DeepTutor’s foundational layer** — the shared personalization substrate and core tutoring pipelines (question generation, solving, RAG, and memory) that the report analyzes. It is the codebase referenced for experiments in the report.

## Personalized Agentic Tutoring (Foundational Layer)

The foundational layer couples **static knowledge grounding** (RAG over multimodal KBs), **dynamic personal memory** (Trace Forest and learner profile), and closed-loop **personalized problem solving** and **question generation**, with a shared learner record feeding both sides.

<p align="center">
  <img src="assets/figs/full-pipe-0316.png" alt="DeepTutor foundational layer" width="920" />
</p>
<p align="center"><em>Overview of the foundational layer: mixed retrieval over the knowledge base, trace-based personalization, multi-stage solve (left) and dual-loop question generation (right).</em></p>


| Area         | Path                   | Role                                                    |
| ------------ | ---------------------- | ------------------------------------------------------- |
| **Solve**    | `src/agents/solve/`    | Plan → ReAct → Write with tools and citations           |
| **Question** | `src/agents/question/` | Idea loop + generation loop; topic and exam-mimic modes |
| **RAG**      | `src/services/rag/`    | LlamaIndex, LightRAG, RAG-Anything backends             |
| **Memory**   | `src/personalization/` | Trace Forest, reflection / summary / weakness agents    |


<details>
<summary><strong>Installation</strong> (Python 3.10+)</summary>

```bash
git clone <repo-url>
cd DeepTutor
git checkout eval

# Option A: venv
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Option B: conda
conda create -n deeptutor python=3.10 -y
conda activate deeptutor

pip install -r requirements.txt
cp .env.example .env
```

**Environment variables:**

| Key | Description |
|-----|-------------|
| `LLM_BINDING` | Provider: `openai`, `anthropic`, `deepseek`, etc. |
| `LLM_MODEL` | Model name, e.g. `gpt-4o` |
| `LLM_API_KEY` | API key |
| `LLM_HOST` | API endpoint |
| `EMBEDDING_BINDING` | Embedding provider |
| `EMBEDDING_MODEL` | Embedding model name |
| `EMBEDDING_API_KEY` | Embedding API key |
| `EMBEDDING_HOST` | Embedding endpoint |
| `EMBEDDING_DIMENSION` | Vector dimension |

Optional: `SEARCH_PROVIDER`, `SEARCH_API_KEY` (see `.env.example`).

</details>



## CLI Usage

### Interactive launcher

```bash
python start.py
```

### Solve CLI

```bash
python -m src.agents.solve.cli "What is linear convolution?" --kb Calculus
python -m src.agents.solve.cli "What is linear convolution?" --detailed
python -m src.agents.solve.cli -i --language zh
```

### Question CLI

```bash
python src/agents/question/cli.py
```

Topic mode and exam-mimic mode. Memory hooks run when launched via `start.py`.

---

## Evaluation

Evaluation follows three axes: **TutorBench construction**, **first-person interactive tutoring evaluation**, and **general problem-solving transfer**, plus **component ablations**. Below, each figure is paired with the scripts that implement the same logic.

### TutorBench: benchmark construction

University-level sources are indexed into per-domain knowledge bases; each KB yields learner profiles (beginner / intermediate / advanced), source-grounded knowledge gaps (misconception, incomplete, missing), and interactive tasks assembled with rejection sampling. The pipeline writes structured entries under the default output root.

<p align="center">
  <img src="assets/figs/bench-0315.png" alt="TutorBench construction pipeline" width="920" />
</p>
<p align="center"><em>TutorBench: profile + knowledge gaps + interactive task per entry.</em></p>

**Step 1 — generate scopes, profiles, and entries** (maps to benchmark construction in the report):

```bash
python benchmark/pipeline/step1_generate_entries.py \
  --kb-names "Calculus,LinearAlgebra" \
  --kb-dir data/knowledge_bases
```

Default output root: `benchmark/data/bench_pipeline/`. Entries: `entries/<kb_name>/profiles/<profile_id>/entries.jsonl`.

<details>
<summary>Step 1 CLI flags</summary>

| Flag | Description |
|------|-------------|
| `--kb-names` | Comma-separated KB names (required) |
| `--kb-dir` | KB root (default: `data/knowledge_bases`) |
| `--config` | Path to `benchmark/config/benchmark_config.yaml` |
| `--output-root` | Override pipeline output root |
| `--concurrency` | Parallel profile tasks (default: 6) |
| `--kb-concurrency` | Parallel KB tasks (default: 6) |

</details>


### First-person interactive evaluation

A student simulator is initialized from a TutorBench entry (profile, gaps, task); the tutor and simulator exchange multi-turn dialogue; transcripts are scored by an independent LLM judge on solve-side and practice-side rubrics.

<p align="center">
  <img src="assets/figs/eval-0315.png" alt="First-person interactive evaluation protocol" width="920" />
</p>
<p align="center"><em>First-person protocol: simulator ↔ tutor dialogue; traces → personalized rubrics.</em></p>

**Step 2 — run simulations** (generate transcripts per backend):

```bash
python benchmark/pipeline/step2_generate_transcripts.py \
  --kb-names "Calculus,LinearAlgebra" \
  --backends "deep_tutor,mock,cot"
```

### Table 1 — Main results on TutorBench (interactive evaluation)

<p align="center">
  <img src="assets/figs/table-1.png" alt="Table 1: Main results on TutorBench" width="820" />
</p>


**Step 3 — evaluate transcripts** (LLM-as-judge; metrics align with Table 1 below):

```bash
python benchmark/pipeline/step3_evaluate_transcripts.py \
  --kb-names "Calculus,LinearAlgebra" \
  --backends "deep_tutor,mock,cot"
```

<details>
<summary>Step 2 CLI flags</summary>

| Flag | Description |
|------|-------------|
| `--kb-names` | Comma-separated KB names (required) |
| `--output-root` | Pipeline output root |
| `--backends` | `mock`, `cot`, `self_refine`, `react`, `deep_tutor`, `deep_tutor_no_rag`, `deep_tutor_no_memory`, `deep_tutor_no_rag_memory` |
| `--max-turns` | Max student turns per session (default: 30) |
| `--language` | `en` or `zh` |
| `--model` | Override `LLM_MODEL` for simulation |
| `--concurrency` / `--backend-concurrency` | Parallelism |
| `--force` | Rerun sessions even if present |
| `-v` / `--verbose` | Dialogue logs |

</details>


<details>
<summary>Step 3 CLI flags</summary>

| Flag | Description |
|------|-------------|
| `--kb-names` | Comma-separated KB names (required) |
| `--backends` | Backends to score |
| `--model` | Override judge model |
| `--temperature` | Judge temperature (default: 0.2) |
| `--concurrency` | Parallel transcript evaluations |
| `--skip-turns` | Turn-level metrics off (turn count only) |
| `--force` | Overwrite existing eval JSON |

</details>


<details>
<summary>Multi-model Step 2 + fixed judge (Step 3)</summary>

```bash
python -m benchmark.pipeline.run_step2_multimodel_then_step3_fixed_judge \
  --kb-names "Calculus" \
  --step2-models "gpt-4o,deepseek-v3" \
  --judge-model "gpt-4o" \
  --backends "deep_tutor,mock"
```

</details>

<details>
<summary>Standalone transcript evaluation & simulator tools</summary>

```bash
python -m benchmark.evaluation.run --transcript path/to/transcript.json
python -m benchmark.evaluation.run --transcript-dir path/to/transcripts/
```

Simulator workspace tools: `solve_question()`, `generate_questions()`, `submit_answers()` — see [benchmark/simulation/USE_TOOL.md](benchmark/simulation/USE_TOOL.md).

</details>

### Table 2 — General problem-solving (pass@1)

<p align="center">
  <img src="assets/figs/table-2.png" alt="Table 2: General problem-solving pass@1 scores" width="820" />
</p>


**Reproduce isolated solve benchmarks** (this branch):

```bash
python benchmark/iso_solve/run_benchmark.py --benchmark math --mode pipeline --limit 50
python benchmark/iso_solve/run_benchmark.py --benchmark gpqa --mode direct --limit 100
```

<details>
<summary><code>run_benchmark.py</code> flags</summary>

| Flag | Description |
|------|-------------|
| `--benchmark` | `math`, `gpqa`, `aime25`, `hle`, `gaia`, `livebench`, `aalcr`, `super_gpqa` |
| `--mode` | `direct` or `pipeline` |
| `--limit` | Number of problems |
| `--dry-run` | Preview only |

See [benchmark/iso_solve/README.md](benchmark/iso_solve/README.md) for dataset-specific options.

</details>



### Ablation study

RAG and Memory are ablated in the lower block of Table 1; the radar chart summarizes per-metric drops. Removing **RAG** hits grounding and source faithfulness hardest (−24.7% on GND); removing **Memory** hits personalization and fitness most (−8.1% on PER).

<p align="center">
  <img src="assets/figs/ablation-0316.png" alt="Ablation radar — w/o RAG and w/o Memory" width="720" />
</p>
<p align="center"><em>Ablation: full DeepTutor (black) vs. w/o RAG (left) and w/o Memory (right). Labels mark the largest drops.</em></p>

Ablation is reproduced by running Step 2 + Step 3 with all four backend variants:

```bash
# Step 2: generate transcripts for full and ablated variants
python benchmark/pipeline/step2_generate_transcripts.py \
  --kb-names "Calculus,LinearAlgebra" \
  --backends "deep_tutor,deep_tutor_no_rag,deep_tutor_no_memory,deep_tutor_no_rag_memory"

# Step 3: evaluate all variants with the same judge
python benchmark/pipeline/step3_evaluate_transcripts.py \
  --kb-names "Calculus,LinearAlgebra" \
  --backends "deep_tutor,deep_tutor_no_rag,deep_tutor_no_memory,deep_tutor_no_rag_memory"
```

The per-backend breakdown in `manifests/step3_summary.json` directly yields the ablation rows in Table 1.

---

## Repository structure

```text
DeepTutor/
├── DeepTutor_arxiv-7/         # Technical report (LaTeX)
├── assets/figs/               # Figures used in README / paper
├── benchmark/
│   ├── config/
│   ├── data_generation/
│   ├── evaluation/
│   ├── iso_solve/
│   ├── pipeline/
│   ├── prompts/
│   ├── simulation/
│   └── tools/
├── config/
├── src/
├── tests/
├── start.py
├── .env.example
├── pyproject.toml
└── requirements.txt
```

## License

Apache-2.0