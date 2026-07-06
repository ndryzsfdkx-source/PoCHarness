# Terminology

Paper term ↔ code identifier map. Serialized identifiers (trajectory logs,
evaluator reports) are parsed by fixed name and were left unrenamed.

## System and roles

| Paper term | In this repo |
|---|---|
| PoCHarness | this package |
| PoC Solver (Agent A) | `PoCSolverAgent` (`secb/harness/agent.py`); role `a` |
| Synthesis Helper (Agent C) | `SynthesisHelperAgent` (`secb/harness/agent.py`); role `c`; managed-agent name `helper` |
| PoC Reviewer (Agent B) | `PoCReviewerAgent` (`secb/review/reviewer.py`); role `b` |
| Crash-only Evaluator | evaluator `loose` |
| Path-aware Evaluator | evaluator `caller` |
| Function-level Evaluator | evaluator `semantic` |
| Source-location Evaluator | evaluator `strict` |

## Modules

| Paper concept | Module |
|---|---|
| Solver + Synthesis Helper | `secb/harness/` |
| PoC Reviewer | `secb/review/` |
| Evaluation algorithm (four-evaluator oracle) | `secb/sanitizer/` |
| Per-role cost accounting | `secb/cost_budget/` |
| Trajectory formatting | `secb/trajectory/` |

(paths relative to `vendor/smolagents/src/smolagents/`)

The runner that *invokes* the evaluation algorithm above (CLI plumbing, one
report file per evaluator) is `vendor/sec-bench-evaluator/secb/evaluator/` — a
vendored, extended copy of SEC-bench's own evaluation harness, not part of
the `smolagents.secb` namespace. See its `NOTICE` entry for attribution.

## Config keys (TOML)

| Paper concept | TOML key |
|---|---|
| Enable A+B+C scaffold | `[agent.pocharness]` |
| Synthesis Helper settings | `[agent.pocharness.helper]` |
| PoC Reviewer settings | `[agent.pocharness.reviewer]` |
| A's own pre-submission self-check (≠ B's review) | `[agent.pocharness.finalization_guard]` |
| PoC Reviewer's evidence-grounded crash gate | `[agent.pocharness.reviewer.evidence_policy]` |

A-only runs omit `[agent.pocharness]` entirely.

## Prompt files

| Paper concept | File |
|---|---|
| A+B+C task prompt | `prompts/poc-desc-pocharness.j2` |
| A-only task prompt | `prompts/poc-desc-floor.j2` (includes `poc-desc.j2`) |
| Synthesis Helper prompt | `secb/harness/prompts/synthesis_helper*.j2` |
| PoC Reviewer prompt | `secb/review/prompts/poc_reviewer*.j2` |

## Notes

Log filenames, evaluator tokens, role keys, verdict enums, and tool names in
the code and logs are left as-is rather than renamed to paper terms, since
they're parsed by fixed name at runtime. The package import name also stays
`smolagents` (this project's contribution is the `smolagents.secb.*`
namespace, not a fork of the whole library), and the
`hwiwonlee/secb.eval.x86_64.*` Docker image prefix is upstream SEC-bench's
own naming.
