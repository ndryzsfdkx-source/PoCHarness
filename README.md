<div align="center">
  <img src="assets/banner.svg" alt="PoCHarness — Engineering Agent Harnesses for Agentic Vulnerability PoC Generation, ICSE 2027" width="100%">

  <br>

  ![paper](https://img.shields.io/badge/paper-ICSE%202027-blue)
  ![license](https://img.shields.io/badge/license-Apache--2.0-green)
</div>

<br>

> "Crash is not enough."

Evaluating an agent-generated proof-of-concept by whether the target program
merely *crashes* overcounts: a sanitizer will fire on any number of
off-target faults the agent stumbles into along the way. A PoC only counts if
it reproduces the crash the vulnerability description actually names — the
same crash **type**, at the same **location**. PoCHarness is an agent harness
built around that stricter bar: a PoC Solver that generates candidate
exploits, a Synthesis Helper it can delegate to, and a PoC Reviewer that
gates submissions against evidence before they're accepted.

## Overview

<div align="center">
  <img src="assets/pocharness_overview.png" alt="PoCHarness overview: a Solver harness generates a PoC candidate, optionally delegating to a Synthesis Helper, then a Reviewer harness verifies the submission against observed crash evidence before accepting it or rejecting it back to the Solver with feedback." width="100%">
</div>

<p align="right"><sub><a href="assets/pocharness_overview.pdf">vector PDF</a></sub></p>

The Solver drives exploit generation from the vulnerability description
alone; it can delegate to the Synthesis Helper, which returns a manifest the
Solver folds back into its own attempt. Every candidate PoC then passes
through the Reviewer, which grounds its accept/reject decision in observed
crash evidence rather than the Solver's self-report, and can send a
submission back with concrete feedback instead of a bare rejection.

## The overestimation, quantified

Full 300-instance `poc-desc` split, GPT-5.5:

| | Result |
|---|---|
| Crash-only evaluation | **82.7%** (248/300) — overcounts |
| Source-location baseline (Solver alone) | **43.3%** (130/300) |
| Source-location, +PoCHarness | **50.7%** (152/300) — **+22 instances, +16.9% relative** |

The gap between the first row and the third is the paper's central claim:
naive crash-only grading is claiming success on instances that a
location-aware grader rejects.

### Full four-grader results

Reproduced directly from the released report files (see
[Published results and artifacts](#published-results-and-artifacts)), not
hand-transcribed:

| Result | Crash-only | Path-aware | Function-level | Source-location |
|---|---|---|---|---|
| GPT-5.5 / solver-only | 248 | 156 | 137 | 130 |
| GPT-5.5 / PoCHarness | 251 | 182 | 159 | 152 |
| GPT-5.4-mini / solver-only | 32 | 21 | 20 | 18 |
| GPT-5.4-mini / PoCHarness | 37 | 27 | 23 | 21 |

See [Published results and artifacts](#published-results-and-artifacts)
before citing these as a clean, uncaveated table.

## Target-aware evaluators

Grading gets progressively stricter about *what counts as the right crash*
— not each implying the previous:

| Evaluator | Checks |
|---|---|
| Crash-only | The target sanitizer fires at all |
| Path-aware | The crash occurs along a plausible call path |
| Function-level | The crash occurs in the named function |
| Source-location | The crash occurs at the named source location |

This four-grader oracle is this project's extension of SEC-bench's
evaluation harness, which ships a single pass/fail oracle upstream.

## Published results and artifacts

The per-instance artifacts and evaluation reports are available in the
[Zenodo results record (restricted access)](https://zenodo.org/records/21194495?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjhiZTdhZTQ5LTBmMjgtNDEzMC1hNGE1LTdiZGZhZDQzYzgyNyIsImRhdGEiOnt9LCJyYW5kb20iOiIwNGRjZWQwZGEyZDU3NzZhMDNhNzUyZjRlODZkZjk4MiJ9.LfScpZ_HJDd2hb-Bg5yEZjZa79444AvlADHAc3X2gQcWRSefPkFIin-IfOwdOsBWyhRF9chdHM6rsjxLfRt3GA).
Open `SUMMARY.csv` for the raw pass/fail table.

## Inspecting published results

No rerun or API cost is required. Download the results tar, then run:

```bash
mkdir pocharness-results-anon
tar -xf pocharness-results-anon.tar -C pocharness-results-anon

python src/pocharness/analyze_run.py \
  --eval-dir pocharness-results-anon/gpt-5.5/pocharness \
  --output analysis.md
```

This reads the four grader counts directly from the shipped reports. Swap
`gpt-5.5/pocharness` for `gpt-5.5/solver-only`,
`gpt-5.4-mini/solver-only`, or
`gpt-5.4-mini/pocharness` for the other results.
For a fresh run, point `--eval-dir` to its `runs/main/.../eval/...` directory.

## Repository structure

```
configs/                     # TOML configs for the 4 reported runs
src/pocharness/               # orchestrator + analysis CLI
  run_secbench_poc.py         #   generate/eval/analyze pipeline entrypoint
  analyze_run.py               #   grader-count and trajectory readout
vendor/
  smolagents/                 # vendored agent framework fork
    src/smolagents/secb/      #   this project's Solver/Helper/Reviewer/grading code
  sec-bench-evaluator/        # vendored SEC-bench evaluator + four-grader oracle
environment.yml
SETUP.md
TERMINOLOGY.md
LICENSE
NOTICE
```

## Reproducing from scratch

Fresh all-300-instance runs require Docker, live API calls, and substantial
cost. See [`SETUP.md`](SETUP.md) for setup, commands, and tests.

## Terminology

Paper terms (PoC Solver / Synthesis Helper / PoC Reviewer, the four
graders) map onto specific code identifiers, modules, and config keys —
see [`TERMINOLOGY.md`](TERMINOLOGY.md).

## Citation

```bibtex
PLACEHOLDER
```

## License

Apache-2.0 — see [`LICENSE`](LICENSE). Third-party provenance (vendored
smolagents fork, vendored SEC-bench evaluator) is in [`NOTICE`](NOTICE).
