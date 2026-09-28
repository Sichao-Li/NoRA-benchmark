"""Batched text-pair scoring with per-clip caching."""

from itertools import product

import numpy as np

from nora._scoring.embedding_backend import SentenceTransformerBackend


class CachedCrossEncoder(SentenceTransformerBackend):
    def __init__(self, model, model_name):
        super().__init__(model_name=model_name, allow_fallback=False)
        self._model = model
        self._pairs = {}

    def prepare(self, prediction, reference):
        pairs = set()
        for kind in ("action", "fact", "reason"):
            def texts(instance):
                if kind == "action":
                    return {graph.action_text for graph in instance.action_graphs}
                return {node.text for graph in instance.action_graphs
                        for node in getattr(graph, kind + "s")}
            pairs.update(product(texts(prediction), texts(reference)))
        ordered = sorted(pairs)
        self._pairs.clear()
        if ordered:
            values = self._model.predict(ordered, batch_size=32, show_progress_bar=False,
                                         convert_to_numpy=True)
            # Normalize values independently so batch composition cannot change
            # the score scale.
            self._pairs.update((pair, float(self._normalize_cross_encoder_scores(
                np.asarray([value], dtype=float))[0]))
                for pair, value in zip(ordered, values))

    def similarity(self, text_a, text_b):
        key = (text_a, text_b)
        if key not in self._pairs:
            self._pairs[key] = super().similarity(text_a, text_b)
        return self._pairs[key]
