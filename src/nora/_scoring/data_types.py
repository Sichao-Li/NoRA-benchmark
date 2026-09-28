from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

NodeType = Literal["fact", "reason", "action"]
EdgeType = Literal["supports", "motivates"]


class ValidationError(ValueError):
    """Raised when an action-rooted support graph payload is malformed."""


def _require_non_empty(value: str, field_name: str) -> str:
    text = value.strip()
    if not text:
        raise ValidationError(f"{field_name} must be a non-empty string.")
    return text


def _ensure_mapping(payload: Mapping[str, Any] | Any, context: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValidationError(f"{context} must be an object.")
    return payload


@dataclass(frozen=True, slots=True)
class FactNode:
    id: str
    text: str
    node_type: Literal["fact"] = "fact"

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_non_empty(self.id, "fact.id"))
        object.__setattr__(self, "text", _require_non_empty(self.text, "fact.text"))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactNode":
        data = _ensure_mapping(payload, "fact")
        return cls(
            id=str(data.get("id", "")).strip(),
            text=str(data.get("text", "")).strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text}


@dataclass(frozen=True, slots=True)
class ReasonNode:
    id: str
    text: str
    normative_label: str | None = None
    node_type: Literal["reason"] = "reason"

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_non_empty(self.id, "reason.id"))
        object.__setattr__(self, "text", _require_non_empty(self.text, "reason.text"))
        if self.normative_label is not None:
            label = self.normative_label.strip()
            object.__setattr__(self, "normative_label", label or None)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReasonNode":
        data = _ensure_mapping(payload, "reason")
        raw_label = data.get("normative_label")
        label = None if raw_label is None else str(raw_label)
        return cls(
            id=str(data.get("id", "")).strip(),
            text=str(data.get("text", "")).strip(),
            normative_label=label,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"id": self.id, "text": self.text}
        if self.normative_label is not None:
            payload["normative_label"] = self.normative_label
        return payload


@dataclass(frozen=True, slots=True)
class ActionNode:
    id: str
    text: str
    node_type: Literal["action"] = "action"

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_non_empty(self.id, "action.id"))
        object.__setattr__(self, "text", _require_non_empty(self.text, "action.text"))

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text}


@dataclass(frozen=True, slots=True)
class Edge:
    src: str
    dst: str
    type: EdgeType

    def __post_init__(self) -> None:
        object.__setattr__(self, "src", _require_non_empty(self.src, "edge.src"))
        object.__setattr__(self, "dst", _require_non_empty(self.dst, "edge.dst"))
        if self.type not in {"supports", "motivates"}:
            raise ValidationError(f"Unsupported edge type: {self.type!r}")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Edge":
        data = _ensure_mapping(payload, "edge")
        return cls(
            src=str(data.get("src", "")).strip(),
            dst=str(data.get("dst", "")).strip(),
            type=str(data.get("type", "")).strip(),  # type: ignore[arg-type]
        )

    def to_dict(self) -> dict[str, Any]:
        return {"src": self.src, "dst": self.dst, "type": self.type}


@dataclass(frozen=True, slots=True)
class ActionGraph:
    action_id: str
    action_text: str
    facts: tuple[FactNode, ...] = field(default_factory=tuple)
    reasons: tuple[ReasonNode, ...] = field(default_factory=tuple)
    edges: tuple[Edge, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_id", _require_non_empty(self.action_id, "action_graph.action_id"))
        object.__setattr__(self, "action_text", _require_non_empty(self.action_text, "action_graph.action_text"))
        object.__setattr__(self, "facts", tuple(self.facts))
        object.__setattr__(self, "reasons", tuple(self.reasons))
        object.__setattr__(self, "edges", tuple(self.edges))
        self._validate_unique_ids()
        self._validate_edges()

    @property
    def action_node(self) -> ActionNode:
        return ActionNode(id=self.action_id, text=self.action_text)

    @property
    def node_lookup(self) -> dict[str, FactNode | ReasonNode | ActionNode]:
        lookup: dict[str, FactNode | ReasonNode | ActionNode] = {self.action_id: self.action_node}
        lookup.update({node.id: node for node in self.facts})
        lookup.update({node.id: node for node in self.reasons})
        return lookup

    @property
    def fact_texts(self) -> tuple[str, ...]:
        return tuple(node.text for node in self.facts)

    @property
    def reason_texts(self) -> tuple[str, ...]:
        return tuple(node.text for node in self.reasons)

    def _validate_unique_ids(self) -> None:
        seen: set[str] = set()
        for node_id in [self.action_id, *[fact.id for fact in self.facts], *[reason.id for reason in self.reasons]]:
            if node_id in seen:
                raise ValidationError(f"Duplicate node id {node_id!r} in action graph {self.action_id!r}.")
            seen.add(node_id)

    def _validate_edges(self) -> None:
        lookup = self.node_lookup
        seen_edges: set[tuple[str, str, str]] = set()
        for edge in self.edges:
            if edge.src not in lookup:
                raise ValidationError(f"Edge source {edge.src!r} is not defined in action graph {self.action_id!r}.")
            if edge.dst not in lookup:
                raise ValidationError(f"Edge destination {edge.dst!r} is not defined in action graph {self.action_id!r}.")
            src_type = lookup[edge.src].node_type
            dst_type = lookup[edge.dst].node_type
            if edge.type == "supports" and not (src_type == "fact" and dst_type == "reason"):
                raise ValidationError(
                    "supports edges must connect fact -> reason; "
                    f"got {src_type} -> {dst_type} in action graph {self.action_id!r}."
                )
            if edge.type == "motivates" and not (src_type == "reason" and dst_type == "action"):
                raise ValidationError(
                    "motivates edges must connect reason -> action; "
                    f"got {src_type} -> {dst_type} in action graph {self.action_id!r}."
                )
            signature = (edge.src, edge.dst, edge.type)
            if signature in seen_edges:
                raise ValidationError(f"Duplicate edge {signature!r} in action graph {self.action_id!r}.")
            seen_edges.add(signature)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActionGraph":
        data = _ensure_mapping(payload, "action_graph")
        facts_raw = data.get("facts", [])
        reasons_raw = data.get("reasons", [])
        edges_raw = data.get("edges", [])
        if not isinstance(facts_raw, Sequence) or isinstance(facts_raw, (str, bytes)):
            raise ValidationError(f"facts must be a list in action graph {data.get('action_id', '<unknown>')!r}.")
        if not isinstance(reasons_raw, Sequence) or isinstance(reasons_raw, (str, bytes)):
            raise ValidationError(f"reasons must be a list in action graph {data.get('action_id', '<unknown>')!r}.")
        if not isinstance(edges_raw, Sequence) or isinstance(edges_raw, (str, bytes)):
            raise ValidationError(f"edges must be a list in action graph {data.get('action_id', '<unknown>')!r}.")
        return cls(
            action_id=str(data.get("action_id", "")).strip(),
            action_text=str(data.get("action_text", "")).strip(),
            facts=tuple(FactNode.from_dict(item) for item in facts_raw),
            reasons=tuple(ReasonNode.from_dict(item) for item in reasons_raw),
            edges=tuple(Edge.from_dict(item) for item in edges_raw),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_text": self.action_text,
            "facts": [node.to_dict() for node in self.facts],
            "reasons": [node.to_dict() for node in self.reasons],
            "edges": [edge.to_dict() for edge in self.edges],
        }


@dataclass(frozen=True, slots=True)
class ReasoningInstance:
    instance_id: str
    action_graphs: tuple[ActionGraph, ...]
    chosen_action_id: str | None = None
    chosen_action_text: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "instance_id", _require_non_empty(self.instance_id, "instance.instance_id"))
        object.__setattr__(self, "action_graphs", tuple(self.action_graphs))
        if self.chosen_action_id is not None:
            chosen_action_id = self.chosen_action_id.strip()
            object.__setattr__(self, "chosen_action_id", chosen_action_id or None)
        if self.chosen_action_text is not None:
            chosen_action_text = self.chosen_action_text.strip()
            object.__setattr__(self, "chosen_action_text", chosen_action_text or None)
        seen_action_ids: set[str] = set()
        for graph in self.action_graphs:
            if graph.action_id in seen_action_ids:
                raise ValidationError(f"Duplicate action_id {graph.action_id!r} in instance {self.instance_id!r}.")
            seen_action_ids.add(graph.action_id)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReasoningInstance":
        data = _ensure_mapping(payload, "instance")
        graphs_raw = data.get("action_graphs", [])
        if not isinstance(graphs_raw, Sequence) or isinstance(graphs_raw, (str, bytes)):
            raise ValidationError("action_graphs must be a list.")
        return cls(
            instance_id=str(data.get("instance_id", "")).strip(),
            action_graphs=tuple(ActionGraph.from_dict(item) for item in graphs_raw),
            chosen_action_id=None if data.get("chosen_action_id") is None else str(data.get("chosen_action_id")),
            chosen_action_text=None if data.get("chosen_action_text") is None else str(data.get("chosen_action_text")),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "instance_id": self.instance_id,
            "action_graphs": [graph.to_dict() for graph in self.action_graphs],
        }
        if self.chosen_action_id is not None:
            payload["chosen_action_id"] = self.chosen_action_id
        if self.chosen_action_text is not None:
            payload["chosen_action_text"] = self.chosen_action_text
        return payload
