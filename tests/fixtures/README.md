# Test fixtures

`annotations/` contains two sanitized HumanGold-style annotations with anonymous
clip IDs. `demo_pred_generic.json` provides a weak prediction for regression
tests. These files are not benchmark results and contain no source media.

The demo's reference and grounded prediction live in `src/nora/assets/` so the
CLI and tests use the same fixtures.
