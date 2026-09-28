# Contributing

Install the test dependencies and run the checks:

```bash
uv sync --locked --extra test
make check
```

Tests use synthetic or sanitized fixtures and must not make paid API calls.
For a semantic-scoring check, install `uv sync --locked --extra scorer`.

## Package layout

- `src/nora/`: data/media loading, validation, inference, reconstruction, evaluation, and CLI.
- `src/nora/_scoring/`: private scoring formulas and graph operations.
- `src/nora/assets/`: synthetic demo, prediction prompts, and human-reviewed test annotations.
- `tests/`: regression tests and fixtures.
- `docs/`: model integration and scoring guides.

Keep one implementation of each scoring rule. Changes to parsing or scoring
need regression tests; changes to metric semantics need a new protocol version.
Add model integrations through the callback interface before adding dependencies
or another runner. The supported Python interface is `nora`; scoring internals
are private. Export supported functions in `nora.__all__`, use descriptive verbs,
and keep CLI options, function parameters, help text, and examples consistent.
Preserve published metric and annotation-field names.

Keep reference answers out of inference and reconstruction. Never commit
credentials, media, checkpoints, or inference dumps. Use environment variables
for credentials and synthetic examples in tests.
