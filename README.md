# NoRA

**Evaluating Grounded Reasonableness in Visual First-person Normative Action Reasoning**

[Paper](https://arxiv.org/abs/2606.04806) |
[Dataset](https://huggingface.co/datasets/MINTLABJHUANU/NoRA) |
[Model guide](docs/models.md) |
[Scoring guide](docs/evaluation.md)

NoRA evaluates the actions a model proposes, the facts it observes, and the
reasons connecting them. It includes 190 human-annotated examples
(HumanGold) and 1,230 model-annotated training examples (LLMSilver).

Use this package to evaluate your own model.

## Quick start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Sichao-Li/NoRA-benchmark.git
cd NoRA-benchmark
uv sync --locked
uv run nora demo
```

This creates a local `.venv/` and runs a small synthetic example.
The demo needs no GPU or API key; its scores are **not benchmark results**.

## Evaluate predictions

Save one prediction per line in JSONL format. Each prediction must use a test
`clip_id` and contain facts, reasons, and actions; see the
[format examples](docs/models.md#prediction-formats).

```bash
uv sync --locked --extra scorer
uv run nora validate predictions.jsonl
uv run --extra scorer nora evaluate \
  --predictions predictions.jsonl --output runs/my-model
```

The package includes the human test references. Only the pinned
semantic scorer needs downloading; once cached, add `--offline`. CPU scoring is
supported. Use `--references path/to/reference.jsonl` for a different reference
file and `--progress` for clip-level progress.

All submitted candidate actions are scored, including actions without support.
The [scoring guide](docs/evaluation.md#reproducibility) describes the evaluation
settings to record when reporting or reproducing results.

Each run saves scores, per-example results, and coverage. Invalid and missing
predictions are recorded separately, not assigned zero scores. Incorrect
predictions that meet the format requirements still count.

To compare models on the same valid examples:

```bash
uv run nora compare runs/model-a runs/model-b
```

Always report coverage alongside scores: a comparison on a valid subset is not
a full-test result. See the [scoring guide](docs/evaluation.md) for the metrics,
output files, and comparison requirements.

## Use your model

| Input | How to use it |
| --- | --- |
| Saved predictions | Evaluate annotation or graph JSONL files. |
| A local or custom model | Pass an inference function to `nora.predict`. |
| An image- or video-capable API | Use `nora predict` with a compatible Chat Completions endpoint. |

Download the original pre-action inputs, then select the same modality for inference:

```bash
uv run nora download-media --media frames --output data/media --limit 2
uv run nora predict --media frames --media-root data/media \
  --base-url http://localhost:8000/v1 --model YOUR_MODEL \
  --limit 2 --output runs/raw.jsonl
```

Choose `--media video` in both commands for a compatible video model/endpoint.
The [model guide](docs/models.md#python-interface) lists the public Python functions,
their CLI equivalents, and return values. It also covers API keys and text
reconstruction. Raw text must be reconstructed before scoring; dictionaries in
the annotation format can be scored directly.
Reference annotations are never passed to models by the inference interface.

Inference and text reconstruction may incur API fees. Validation, scoring,
tests, and the demo make no paid API calls.

An evaluation with no valid predictions saves diagnostics and exits with failure.
Valid subsets still succeed, with missing and invalid cases reported separately.

## Development

```bash
make check
```

See [Contributing](CONTRIBUTING.md) for the package layout and test workflow.

The project page is a standalone static website in [`website/`](website/README.md).
Run `make website` to preview it locally at `http://127.0.0.1:8766`.

## Citation and license

Please cite the [NoRA paper](https://arxiv.org/abs/2606.04806);
citation metadata is in [CITATION.cff](CITATION.cff).

Code is licensed under [MIT](LICENSE). Dataset annotations are
[CC BY-NC 4.0](LICENSE-DATA).
Source images and videos have separate access and usage terms.
