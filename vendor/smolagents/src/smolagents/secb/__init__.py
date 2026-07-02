"""PoCHarness extensions to upstream smolagents.

All non-upstream code lives under this namespace so the `smolagents/` root
stays close to upstream. Submodules:

- ``sanitizer``: crash parsing and grading
- ``cost_budget``: shared or independent per-role model-cost ledgers
- ``harness``: PoC Solver + Synthesis Helper — write+execute sub-agent that
  produces runnable helper scripts under ``Helpers/`` when the solver hits a
  synthesis-boundary failure — plus solver-side synthesis tools
- ``review``: PoC Reviewer finalization gate — read-only sub-agent that
  reviews the solver's finalization claim and emits an allow/block verdict
- ``trajectory``: shared trajectory formatting and read-only trajectory tools

Everything here is opt-in unless a config explicitly wires it in.
"""
