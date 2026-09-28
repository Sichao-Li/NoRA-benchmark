from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Mapping

from nora._scoring.data_types import ActionGraph, Edge


try:
    import networkx as nx
except ModuleNotFoundError:  # pragma: no cover - exercised indirectly through fallback behavior
    nx = None


@dataclass
class _FallbackDiGraph:
    nodes: dict[str, dict[str, object]] = field(default_factory=dict)
    successors: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    predecessors: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    def add_node(self, node_id: str, **attrs: object) -> None:
        self.nodes[node_id] = attrs

    def add_edge(self, src: str, dst: str, **attrs: object) -> None:
        self.successors[src].add(dst)
        self.predecessors[dst].add(src)


def build_nx_graph(action_graph: ActionGraph) -> object:
    """Convert an ActionGraph into a typed directed graph."""

    graph = nx.DiGraph() if nx is not None else _FallbackDiGraph()
    graph.add_node(action_graph.action_id, node_type="action", text=action_graph.action_text)
    for fact in action_graph.facts:
        graph.add_node(fact.id, node_type="fact", text=fact.text)
    for reason in action_graph.reasons:
        graph.add_node(reason.id, node_type="reason", text=reason.text, normative_label=reason.normative_label)
    for edge in action_graph.edges:
        graph.add_edge(edge.src, edge.dst, type=edge.type)
    return graph


def ancestor_node_ids(action_graph: ActionGraph) -> set[str]:
    """Return all non-action ancestors of the action node."""

    graph = build_nx_graph(action_graph)
    if nx is not None:
        ancestors = set(nx.ancestors(graph, action_graph.action_id))
    else:
        ancestors = set()
        stack = list(graph.predecessors.get(action_graph.action_id, set()))
        while stack:
            node_id = stack.pop()
            if node_id in ancestors:
                continue
            ancestors.add(node_id)
            stack.extend(graph.predecessors.get(node_id, set()))
    ancestors.discard(action_graph.action_id)
    return ancestors


def edge_signature(edge: Edge) -> tuple[str, str, str]:
    """Stable tuple form for set comparison."""

    return (edge.src, edge.dst, edge.type)


def translate_edge(edge: Edge, node_alignment: Mapping[str, str]) -> tuple[str, str, str] | None:
    """Map a predicted edge into gold node ID space, if both endpoints align."""

    if edge.src not in node_alignment or edge.dst not in node_alignment:
        return None
    return (node_alignment[edge.src], node_alignment[edge.dst], edge.type)
