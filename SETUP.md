# Setup and Reproduction

## Reproducing from scratch

Reruns the 4 reported all-300-instance runs: A-only vs. A+B+C (PoC Solver +
Synthesis Helper + PoC Reviewer), on GPT-5.5 and GPT-5.4-mini — live API
calls and fresh Docker evals, real cost (see below). See `TERMINOLOGY.md`
for paper term ↔ code name mapping.

### Prerequisites

- Docker, running (pulls `hwiwonlee/secb.eval.x86_64.*` images, x86_64).
- conda/miniforge or any Python 3.12 env manager.
- `OPENAI_API_KEY` with access to the reported models.

### 1. Environment

```bash
conda env create -f environment.yml
conda activate pocharness
```

Installs the vendored smolagents fork (editable, `docker`/`litellm`/`secb`
extras) and the vendored SEC-bench evaluator (`vendor/sec-bench-evaluator/`)
— together enough to run the reported configs end to end. `src/pocharness/`
is pure stdlib, needs nothing extra.

To also run the smolagents fork's own pytest suite, use the `test` extra
instead (much heavier: pulls torch, transformers, gradio via
`smolagents[all]`):

```bash
pip install -e "./vendor/smolagents[docker,litellm,secb,test]"
```

**On the vendored evaluator:** `vendor/sec-bench-evaluator/` is a pinned,
MIT-licensed copy of SEC-bench's own evaluation harness, extended with the
four-evaluator PoC oracle this paper's results depend on (upstream only has a
single pass/fail oracle). See `NOTICE` for exact provenance.
`SECBENCH_EVAL_ROOT` overrides the path if you want to point at a different
checkout.

### 2. The four reported runs

```bash
python src/pocharness/run_secbench_poc.py --config configs/all300_solver_only_gpt55.toml
python src/pocharness/run_secbench_poc.py --config configs/all300_pocharness_gpt55.toml
python src/pocharness/run_secbench_poc.py --config configs/all300_solver_only_gpt54mini.toml
python src/pocharness/run_secbench_poc.py --config configs/all300_pocharness_gpt54mini.toml
```

Each runs generation + evaluation (default `--stages`) across 300 instances.
Add `--num-workers N` to parallelize, `--instance-id <id>` for one instance,
or split generate/eval into separate invocations — see `--help`.

**Cost note:** per-instance caps are $2.5 (A-only) or $2.5+$1.0+$2.5 (A+B+C),
times 300 instances.

### 3. Tests (offline)

```bash
cd src/pocharness && pytest tests/ && cd -
cd vendor/smolagents && pytest tests/ && cd -
```
