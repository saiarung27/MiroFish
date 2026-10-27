"""Reify-backed stand-in for the Zep Cloud client.

Selected by ``MIROFISH_MEMORY_BACKEND=reify`` in
:func:`app.utils.zep.get_zep_client`. Every MiroFish service keeps
calling ``client.graph.*`` / ``client.batch.*`` exactly as before;
this module translates those calls onto
:class:`reify.adapters.ReifyZepGraph` (local SQLite stores, one file
per graph, no network, no third-party data license).

Setup (run once for the backend venv)::

    backend/.venv/bin/python -m pip install python-ulid
    PYTHONPATH=/path/to/reify/src   # or: pip install -e /path/to/reify

    MIROFISH_MEMORY_BACKEND=reify
    MIROFISH_REIFY_ROOT=~/.mirofish/reify-graphs   # optional, this is the default
    MIROFISH_REIFY_RELATION_EXTRACTOR=openai:gpt-4o-mini  # optional, off by default

Fidelity notes -- read before assuming Zep parity:

* Results are :class:`AttrDict` objects: attribute access
  (``node.name``) and dict access (``edge.get("fact")``) both work,
  because MiroFish uses both styles across services.
* Node ids are Reify entity slugs (``maya``, ``charlotte``), not
  UUIDs. Callers treat ids as opaque, so this is safe inside one
  backend; do not mix Zep-created and Reify-created graphs.
* Graph existence = the store file exists. ``graph.get`` on a missing
  graph raises :class:`zep_cloud.NotFoundError`, so the existing
  create-reconciliation code paths keep working.
* ``graph.episode.get`` always reports ``processed=True``: Reify
  ingestion is synchronous, there is no async pipeline to poll.
* ``batch.process`` is a no-op that succeeds: episodes are stored at
  ``batch.add`` time. ``batch.get`` reports ``status="processed"``.
* ``graph.search`` is lexical (keyword overlap via Reify recall), not
  semantic, and the ``reranker`` argument is accepted but ignored.
  Zep's cross-encoder/RRF ranking has no local equivalent here.
* ``graph.node.get_edges`` returns outgoing edges only, matching the
  measured Zep Cloud 3.25 behaviour documented in zep_entity_reader.
* ``graph.set_ontology`` is recorded to a JSON sidecar per graph and
  NOT enforced: Reify's indexer does not constrain entities by
  ontology. Same documented limitation as the adapter itself. The
  facade translates the Zep SDK shapes GraphBuilderService actually
  sends (EntityModel/EdgeModel CLASSES and (class, source-targets)
  edge tuples) into plain JSON-safe dicts before recording.
* Edge ids are deterministic ``edge-<sha1(source, relation, target)>``
  and carry no timestamps or episode lists (the adapter does not
  track edge provenance); ``episodes`` is ``[]``.
* Without ``MIROFISH_REIFY_RELATION_EXTRACTOR``, edges are
  deterministic mentions + ``co_occurs`` only. With it, episodes also
  yield LLM-extracted typed edges (see the adapter's relations
  module). Extraction failures never fail the write.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any

from zep_cloud import NotFoundError

from .logger import get_logger

logger = get_logger("mirofish.reify_client")

DEFAULT_REIFY_ROOT = Path.home() / ".mirofish" / "reify-graphs"
_BATCHES_FILENAME = "batches.json"


def _field(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


class AttrDict(dict):
    """A dict that also allows attribute access (``node.name``).

    Missing attributes raise AttributeError so ``hasattr`` /
    ``getattr(..., default)`` checks in callers behave correctly.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


class RawResponse:
    """Minimal stand-in for the SDK's with_raw_response wrapper."""

    def __init__(self, data: Any, headers: dict[str, str] | None = None) -> None:
        self.data = data
        self.headers = headers or {}


def _node_obj(node: dict[str, Any]) -> AttrDict:
    return AttrDict(
        {
            "uuid_": node["uuid"],
            "uuid": node["uuid"],
            "name": node["name"],
            "labels": list(node.get("labels", [])),
            "summary": node.get("summary", ""),
            "attributes": {},
            "created_at": None,
        }
    )


def _edge_obj(edge: dict[str, Any]) -> AttrDict:
    digest = hashlib.sha1(
        f"{edge['source']}\0{edge['relation']}\0{edge['target']}".encode("utf-8")
    ).hexdigest()[:16]
    return AttrDict(
        {
            "uuid_": f"edge-{digest}",
            "uuid": f"edge-{digest}",
            "name": edge["relation"],
            "fact": edge["fact"],
            "source_node_uuid": edge["source"],
            "target_node_uuid": edge["target"],
            "attributes": {},
            "created_at": None,
            "valid_at": None,
            "invalid_at": None,
            "expired_at": None,
            "episodes": [],
            "episode_ids": [],
        }
    )


class _BatchPage:
    """Pagination page with real attributes.

    Deliberately NOT an AttrDict: ``page.items`` must return the item
    list, but on a dict subclass ``.items`` resolves to the builtin
    ``dict.items`` method instead of the stored key. MiroFish reads
    ``getattr(page, "items")`` / ``getattr(page, "batches")``, so plain
    attributes are the correct shape here.
    """

    def __init__(self, *, items: list[Any] | None = None,
                 batches: list[Any] | None = None,
                 next_cursor: int | None = None) -> None:
        self.items = items if items is not None else []
        self.batches = batches if batches is not None else []
        self.next_cursor = next_cursor


class _RawNamespace:
    """``with_raw_response`` shim for node/edge pagination."""

    def __init__(self, fetch: Any) -> None:
        self._fetch = fetch

    def get_by_graph_id(
        self, graph_id: str, *, limit: int = 100, cursor: str | None = None, **_: Any
    ) -> RawResponse:
        # Single-page responses: no next-cursor header, so the paging
        # helpers stop after page one. Graphs here are small.
        return RawResponse(self._fetch(graph_id, limit=limit))


class _EpisodeNamespace:
    def __init__(self, client: "ReifyZepClient") -> None:
        self._client = client

    def get(self, uuid_: str, **_: Any) -> AttrDict:
        episode = self._client._graph_episode(uuid_)
        if episode is None:
            raise NotFoundError(f"episode {uuid_} not found")
        return episode


class _NodeNamespace:
    def __init__(self, client: "ReifyZepClient") -> None:
        self._client = client
        self.with_raw_response = _RawNamespace(
            lambda graph_id, limit: [
                _node_obj(n) for n in self._client._adapter.list_nodes(graph_id)
            ]
        )

    def get(self, uuid_: str, **_: Any) -> AttrDict:
        node = self._client._graph_node(uuid_)
        if node is None:
            raise NotFoundError(f"node {uuid_} not found")
        return node

    def get_edges(self, node_uuid: str, **_: Any) -> list[AttrDict]:
        # Graph-scoped edge reads go through list_edges + filter in the
        # callers; this method is the unscoped Zep shape (source-only).
        return self._client._node_outgoing_edges(node_uuid)


class _EdgeNamespace:
    def __init__(self, client: "ReifyZepClient") -> None:
        self._client = client
        self.with_raw_response = _RawNamespace(
            lambda graph_id, limit: [
                _edge_obj(e) for e in self._client._adapter.list_edges(graph_id)
            ]
        )


def _model_class_to_dict(cls: Any) -> dict[str, Any]:
    """Translate one Zep ontology model to a JSON-safe record.

    GraphBuilderService builds entity/edge types as dynamic Pydantic model
    CLASSES (``type(name, (EntityModel,), attrs)``) and passes the classes
    themselves to ``graph.set_ontology``; the Zep SDK serializes them, but
    the Reify adapter records a JSON sidecar, so a class object here would
    raise ``TypeError: Object of type ModelMetaclass is not JSON
    serializable``. Plain dicts (used by tests and other callers) pass
    through unchanged.
    """
    if isinstance(cls, dict):
        return cls
    fields: dict[str, Any] = {}
    for name, field in getattr(cls, "model_fields", {}).items():
        annotation = getattr(field, "annotation", None)
        fields[name] = {
            "description": getattr(field, "description", None),
            "type": getattr(annotation, "__name__", str(annotation))
            if annotation is not None
            else None,
        }
    return {"description": getattr(cls, "__doc__", None), "attributes": fields}


def _source_target_to_dict(st: Any) -> dict[str, Any]:
    if isinstance(st, dict):
        return st
    return {"source": getattr(st, "source", None), "target": getattr(st, "target", None)}


def _ontology_entities_to_dict(entities: dict[str, Any] | None) -> dict[str, Any]:
    return {name: _model_class_to_dict(cls) for name, cls in (entities or {}).items()}


def _ontology_edges_to_dict(edges: dict[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, value in (edges or {}).items():
        if isinstance(value, dict):
            out[name] = value
            continue
        if isinstance(value, tuple) and len(value) == 2:
            edge_cls, source_targets = value
            record = _model_class_to_dict(edge_cls)
            record["source_targets"] = [
                _source_target_to_dict(st) for st in (source_targets or [])
            ]
            out[name] = record
            continue
        out[name] = _model_class_to_dict(value)
    return out


class _GraphNamespace:
    def __init__(self, client: "ReifyZepClient") -> None:
        self._client = client
        self.episode = _EpisodeNamespace(client)
        self.node = _NodeNamespace(client)
        self.edge = _EdgeNamespace(client)

    def create(self, graph_id: str, name: str | None = None, **_: Any) -> AttrDict:
        self._client._adapter.create_graph(graph_id)
        return AttrDict({"graph_id": graph_id, "name": name or graph_id})

    def get(self, graph_id: str, **_: Any) -> AttrDict:
        if not self._client._graph_exists(graph_id):
            raise NotFoundError(f"graph {graph_id} not found")
        return AttrDict({"graph_id": graph_id, "name": graph_id})

    def delete(self, graph_id: str, **_: Any) -> None:
        self._client._adapter.delete_graph(graph_id)

    def set_ontology(
        self,
        graph_ids: list[str] | None = None,
        entities: dict[str, Any] | None = None,
        edges: dict[str, Any] | None = None,
        **_: Any,
    ) -> None:
        # GraphBuilderService passes Zep SDK shapes, not plain data: entity
        # values are dynamically created EntityModel CLASSES and edge values
        # are (EdgeModel class, [EntityEdgeSourceTarget]) tuples. Those
        # must be translated to plain dicts before the adapter's JSON
        # sidecar write; passing them through verbatim raised "Object of
        # type ModelMetaclass is not JSON serializable" on every real graph
        # build (both Oct 27 E2E tracks hit it at 15% progress).
        for graph_id in graph_ids or []:
            self._client._adapter.set_ontology(
                graph_id,
                {
                    "entities": _ontology_entities_to_dict(entities),
                    "edges": _ontology_edges_to_dict(edges),
                },
            )

    def add(
        self,
        graph_id: str,
        type: str | None = None,
        data: str | None = None,
        created_at: str | None = None,
        source_description: str | None = None,
        metadata: dict[str, Any] | None = None,
        **_: Any,
    ) -> AttrDict:
        episode = self._client._adapter.add_episode(
            graph_id,
            data or "",
            tags=tuple(f"meta:{k}={v}" for k, v in (metadata or {}).items()),
        )
        return AttrDict(
            {
                "uuid_": episode["uuid"],
                "uuid": episode["uuid"],
                "data": data or "",
                "created_at": episode.get("created_at"),
                "processed": True,
            }
        )

    def search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges",
        reranker: str | None = None,
        **_: Any,
    ) -> AttrDict:
        # reranker is accepted for call compatibility and ignored:
        # local search is lexical, not semantic.
        result = self._client._adapter.search(graph_id, query, limit=limit)
        edges = [_edge_obj(e) for e in result.get("edges", [])]
        nodes = [_node_obj(n) for n in result.get("nodes", [])]
        if scope == "edges":
            nodes = []
        elif scope == "nodes":
            edges = []
        return AttrDict({"edges": edges, "nodes": nodes})


class _BatchNamespace:
    def __init__(self, client: "ReifyZepClient") -> None:
        self._client = client

    def create(self, metadata: dict[str, Any] | None = None, **_: Any) -> AttrDict:
        batch_id = f"batch-{uuid.uuid4().hex[:16]}"
        batches = self._client._load_batches()
        batches[batch_id] = {
            "metadata": dict(metadata or {}),
            "graph_id": (metadata or {}).get("graph_id", ""),
            "status": "draft",
            "items": [],
        }
        self._client._save_batches(batches)
        return AttrDict({"batch_id": batch_id, "metadata": dict(metadata or {})})

    def add(self, batch_id: str, items: list[Any], **_: Any) -> list[AttrDict]:
        batches = self._client._load_batches()
        batch = batches.get(batch_id)
        if batch is None:
            raise NotFoundError(f"batch {batch_id} not found")
        graph_id = batch["graph_id"]
        out: list[AttrDict] = []
        for item in items:
            data = _field(item, "data", "") or ""
            metadata = _field(item, "metadata", {}) or {}
            chunk_index = metadata.get("chunk_index", len(batch["items"]))
            episode = self._client._adapter.add_episode(graph_id, data)
            record = {
                "episode_uuid": episode["uuid"],
                "sequence_index": chunk_index,
            }
            batch["items"].append(record)
            out.append(
                AttrDict(
                    {
                        "episode_uuid": episode["uuid"],
                        "sequence_index": chunk_index,
                    }
                )
            )
        self._client._save_batches(batches)
        return out

    def process(self, batch_id: str, **_: Any) -> AttrDict:
        batches = self._client._load_batches()
        batch = batches.get(batch_id)
        if batch is None:
            raise NotFoundError(f"batch {batch_id} not found")
        batch["status"] = "processed"
        self._client._save_batches(batches)
        return AttrDict({"batch_id": batch_id, "status": "processed"})

    def get(self, batch_id: str, **_: Any) -> AttrDict:
        batches = self._client._load_batches()
        batch = batches.get(batch_id)
        if batch is None:
            raise NotFoundError(f"batch {batch_id} not found")
        return AttrDict(
            {
                "batch_id": batch_id,
                "metadata": batch["metadata"],
                "status": batch["status"],
            }
        )

    def list(
        self, limit: int = 100, cursor: int | None = None, **_: Any
    ) -> AttrDict:
        batches = self._client._load_batches()
        ordered = sorted(batches.items())
        start = int(cursor) if cursor is not None else 0
        page = ordered[start : start + limit]
        next_cursor = start + len(page) if start + len(page) < len(ordered) else None
        return _BatchPage(
            batches=[
                AttrDict({"batch_id": bid, "metadata": b["metadata"], "status": b["status"]})
                for bid, b in page
            ],
            next_cursor=next_cursor,
        )

    def list_items(
        self, batch_id: str, limit: int = 100, cursor: int | None = None, **_: Any
    ) -> AttrDict:
        batches = self._client._load_batches()
        batch = batches.get(batch_id)
        if batch is None:
            raise NotFoundError(f"batch {batch_id} not found")
        items = batch["items"]
        start = int(cursor) if cursor is not None else 0
        page = items[start : start + limit]
        next_cursor = start + len(page) if start + len(page) < len(items) else None
        return _BatchPage(
            items=[
                AttrDict(
                    {
                        "episode_uuid": it["episode_uuid"],
                        "sequence_index": it["sequence_index"],
                    }
                )
                for it in page
            ],
            next_cursor=next_cursor,
        )


class ReifyZepClient:
    """A Zep-shaped client backed by local Reify stores."""

    def __init__(self, root_dir: str | Path, *, relation_extractor: Any = None) -> None:
        try:
            from reify.adapters import ReifyZepGraph
        except ImportError as exc:
            raise RuntimeError(
                "Reify is not importable. Install it into the backend "
                "environment (pip install -e /path/to/reify, or add "
                "reify/src to PYTHONPATH) and set "
                "MIROFISH_MEMORY_BACKEND=reify."
            ) from exc
        self._root = Path(root_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        self._adapter = ReifyZepGraph(self._root, relation_extractor=relation_extractor)
        self._lock = threading.Lock()
        self.graph = _GraphNamespace(self)
        self.batch = _BatchNamespace(self)

    def close(self) -> None:
        """Stop the adapter's background loop thread (tests / shutdown)."""
        self._adapter.close()

    # -- internals ---------------------------------------------------------

    def _store_path(self, graph_id: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in graph_id)
        return self._root / f"{safe}.db"

    def _graph_exists(self, graph_id: str) -> bool:
        return self._store_path(graph_id).exists()

    def _graph_episode(self, uuid_: str) -> AttrDict | None:
        # Episodes are memories; the graph is unknown here, so scan the
        # known graph stores. MiroFish always polls episodes of a graph
        # it just built, so the scan is over few files.
        for path in sorted(self._root.glob("*.db")):
            graph_id = path.stem
            episode = self._adapter.get_episode(graph_id, uuid_)
            if episode is not None:
                return AttrDict(
                    {
                        "uuid_": episode["uuid"],
                        "uuid": episode["uuid"],
                        "data": episode["content"],
                        "created_at": episode.get("created_at"),
                        "processed": True,
                    }
                )
        return None

    def _graph_node(self, uuid_: str) -> AttrDict | None:
        for path in sorted(self._root.glob("*.db")):
            node = self._adapter.get_node(path.stem, uuid_)
            if node is not None:
                return _node_obj(node)
        return None

    def _node_outgoing_edges(self, node_uuid: str) -> list[AttrDict]:
        for path in sorted(self._root.glob("*.db")):
            edges = self._adapter.get_node_edges(path.stem, node_uuid)
            if edges:
                return [_edge_obj(e) for e in edges]
        return []

    def _batches_path(self) -> Path:
        return self._root / _BATCHES_FILENAME

    def _load_batches(self) -> dict[str, Any]:
        with self._lock:
            path = self._batches_path()
            if not path.exists():
                return {}
            try:
                data = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                logger.warning("reify batches sidecar unreadable; starting empty")
                return {}
            return data if isinstance(data, dict) else {}

    def _save_batches(self, batches: dict[str, Any]) -> None:
        with self._lock:
            self._batches_path().write_text(json.dumps(batches, indent=2, sort_keys=True))


def resolve_reify_root() -> Path:
    override = os.environ.get("MIROFISH_REIFY_ROOT", "").strip()
    return Path(override).expanduser() if override else DEFAULT_REIFY_ROOT


def _resolve_relation_extractor() -> Any:
    spec = os.environ.get("MIROFISH_REIFY_RELATION_EXTRACTOR", "").strip()
    if not spec:
        return None
    if not spec.startswith("openai:"):
        raise ValueError(
            "MIROFISH_REIFY_RELATION_EXTRACTOR must look like 'openai:gpt-4o-mini'"
        )
    from reify.adapters import LLMRelationExtractor
    from reify.llm import create_llm_provider

    _, model = spec.split(":", 1)
    return LLMRelationExtractor(create_llm_provider(spec), model=model)


def create_reify_client(root_dir: str | Path | None = None) -> ReifyZepClient:
    """Build a Reify-backed client for the configured root dir."""
    return ReifyZepClient(
        root_dir or resolve_reify_root(),
        relation_extractor=_resolve_relation_extractor(),
    )
