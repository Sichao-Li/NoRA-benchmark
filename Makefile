.PHONY: help install install-scorer test demo check

help:
	@printf '%s\n' 'make install        Install the locked local environment' 'make install-scorer Install semantic scoring dependencies' 'make test           Run offline tests' 'make demo           Run the non-benchmark synthetic demo' 'make check          Check CLI, compilation and tests'

install:
	uv sync --locked

install-scorer:
	uv sync --locked --extra scorer

test:
	uv run --locked --extra test pytest -q

demo:
	uv run --locked nora demo

check: test
	uv run --locked nora --help
	uv run --locked python -m compileall -q src
