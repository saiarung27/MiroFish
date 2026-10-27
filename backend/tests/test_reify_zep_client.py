"""Facade tests: the Reify-backed client must satisfy every call shape
MiroFish services use against the Zep Cloud client.

Runs against a temp root dir; no network, no Zep key. The default
(Zep Cloud) path is covered by the last two tests: key still required,
and a dummy key still yields a real Zep client object.
"""

import os

import pytest
from zep_cloud import BatchAddItem, NotFoundError

from app.utils.reify_zep_client import ReifyZepClient
from app.utils.zep import clear_zep_client_cache, get_memory_backend, get_zep_client


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MIROFISH_MEMORY_BACKEND", "reify")
    monkeypatch.setenv("MIROFISH_REIFY_ROOT", str(tmp_path / "graphs"))
    c = get_zep_client()
    assert isinstance(c, ReifyZepClient)
    yield c
    c.close()


def test_backend_selection_and_key_rules(monkeypatch):
    monkeypatch.delenv("MIROFISH_MEMORY_BACKEND", raising=False)
    assert get_memory_backend() == "zep"
    monkeypatch.setenv("MIROFISH_MEMORY_BACKEND", "bogus")
    with pytest.raises(ValueError):
        get_memory_backend()


def test_graph_create_get_delete_roundtrip(client):
    client.graph.create(graph_id="g1", name="G1", description="d")
    got = client.graph.get("g1")
    assert got.graph_id == "g1"
    with pytest.raises(NotFoundError):
        client.graph.get("missing-graph")
    client.graph.delete(graph_id="g1")
    with pytest.raises(NotFoundError):
        client.graph.get("g1")


def test_graph_add_and_episode_processed(client):
    client.graph.create(graph_id="g1")
    episode = client.graph.add(
        graph_id="g1", type="text", data="Maya says 99 dollars a month is too steep.",
        source_description="test", metadata={"source": "test"},
    )
    assert episode.uuid_ and episode.uuid == episode.uuid_
    fetched = client.graph.episode.get(uuid_=episode.uuid_)
    assert fetched.processed is True
    with pytest.raises(NotFoundError):
        client.graph.episode.get(uuid_="no-such-episode")


def test_batch_lifecycle_matches_graph_builder_contract(client):
    client.graph.create(graph_id="g1")
    batch = client.batch.create(
        metadata={"mirofish_operation_id": "op-1", "graph_id": "g1", "chunk_count": 2}
    )
    assert batch.batch_id
    items = [
        BatchAddItem(
            type="graph_episode", graph_id="g1", data="Maya runs a solo crew in Charlotte.",
            data_type="text", source_description="chunk",
            metadata={"mirofish_operation_id": "op-1", "chunk_index": 0},
        ),
        BatchAddItem(
            type="graph_episode", graph_id="g1", data="Darnell worries about shared ZIP codes.",
            data_type="text", source_description="chunk",
            metadata={"mirofish_operation_id": "op-1", "chunk_index": 1},
        ),
    ]
    details = client.batch.add(batch_id=batch.batch_id, items=items)
    assert len(details) == 2
    assert [d.sequence_index for d in details] == [0, 1]
    assert all(d.episode_uuid for d in details)

    client.batch.process(batch_id=batch.batch_id)
    # Zep vocabulary: a processed batch reports "succeeded" (a terminal
    # state for GraphBuilderService's wait loop) plus a progress object.
    summary = client.batch.get(batch_id=batch.batch_id)
    assert summary.status == "succeeded"
    assert summary.progress.percent_complete == 100.0
    assert summary.progress.succeeded_items == 2

    page = client.batch.list_items(batch_id=batch.batch_id, limit=100, cursor=None)
    assert len(page.items) == 2
    assert page.next_cursor is None
    assert all(it.status == "succeeded" for it in page.items)
    assert all(it.source_uuid == it.episode_uuid for it in page.items)

    listing = client.batch.list(limit=100, cursor=None)
    matches = [
        b for b in listing.batches
        if b.metadata.get("mirofish_operation_id") == "op-1"
        and b.metadata.get("graph_id") == "g1"
    ]
    assert len(matches) == 1


def test_raw_pagination_node_edge_shapes(client):
    client.graph.create(graph_id="g1")
    client.graph.add(graph_id="g1", type="text", data="Maya runs a solo crew in Charlotte.")
    client.graph.add(graph_id="g1", type="text", data="Darnell worries about shared ZIP codes.")

    node_page = client.graph.node.with_raw_response.get_by_graph_id("g1", limit=100)
    assert node_page.data
    assert node_page.headers.get("zep-next-cursor") is None
    node = node_page.data[0]
    assert node.uuid_ and node.name and isinstance(node.labels, list)

    edge_page = client.graph.edge.with_raw_response.get_by_graph_id("g1", limit=100)
    assert edge_page.data
    edge = edge_page.data[0]
    assert edge.fact and edge.source_node_uuid and edge.target_node_uuid

    detail = client.graph.node.get(uuid_=node.uuid_)
    assert detail.name == node.name
    with pytest.raises(NotFoundError):
        client.graph.node.get(uuid_="no-such-node")


def test_search_scopes_and_ontology_recorded(client):
    client.graph.create(graph_id="g1")
    # Two entities (Maya, Charlotte) so the episode yields an edge;
    # a single-entity episode correctly yields zero edges.
    client.graph.add(graph_id="g1", type="text",
                     data="Maya says 99 dollars a month is too steep for Charlotte cleaners.")
    client.graph.set_ontology(
        graph_ids=["g1"],
        entities={"Cleaner": {"description": "a cleaning business"}},
        edges=None,
    )
    result = client.graph.search(
        graph_id="g1", query="Maya price", limit=10, scope="edges", reranker="cross_encoder"
    )
    assert result.edges  # lexical search finds the Maya episode's edges
    nodes_only = client.graph.search(
        graph_id="g1", query="Maya", limit=10, scope="nodes", reranker="rrf"
    )
    assert nodes_only.edges == [] and nodes_only.nodes
    recorded = client._adapter.get_ontology("g1")
    assert recorded["entities"]["Cleaner"]["description"] == "a cleaning business"


def test_set_ontology_accepts_zep_model_classes(client):
    """Regression: GraphBuilderService passes EntityModel/EdgeModel CLASSES
    (not dicts) to graph.set_ontology, with edges as (class,
    [EntityEdgeSourceTarget]) tuples. Passing them through verbatim made
    the adapter's json.dumps raise 'ModelMetaclass is not JSON
    serializable' on every real graph build (both Oct 27 E2E tracks
    failed at 15% progress); the facade must translate them to plain
    JSON-safe dicts first."""
    import warnings
    from typing import Optional

    from pydantic import Field
    from zep_cloud import EntityEdgeSourceTarget
    from zep_cloud.external_clients.ontology import EdgeModel, EntityModel, EntityText

    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
    entity_cls = type("BeginnerInvestor", (EntityModel,), {
        "__doc__": "A novice investor.",
        "__annotations__": {"full_name": Optional[EntityText]},
        "full_name": Field(description="The full name.", default=None),
    })
    edge_cls = type("SubscribesTo", (EdgeModel,), {
        "__doc__": "A user subscribes to a service.",
        "__annotations__": {"details": Optional[str]},
        "details": Field(description="Additional details.", default=None),
    })

    client.graph.create(graph_id="g1")
    client.graph.set_ontology(
        graph_ids=["g1"],
        entities={"BeginnerInvestor": entity_cls},
        edges={"SUBSCRIBES_TO": (
            edge_cls,
            [EntityEdgeSourceTarget(source="Person", target="Organization")],
        )},
    )
    recorded = client._adapter.get_ontology("g1")
    entity = recorded["entities"]["BeginnerInvestor"]
    assert entity["description"] == "A novice investor."
    assert entity["attributes"]["full_name"]["description"] == "The full name."
    edge = recorded["edges"]["SUBSCRIBES_TO"]
    assert edge["source_targets"] == [{"source": "Person", "target": "Organization"}]
    assert edge["attributes"]["details"]["description"] == "Additional details."


def test_wait_for_batch_terminates_under_reify(monkeypatch, tmp_path):
    """Regression: the build task's GraphBuilderService._wait_for_batch
    polls batch.get until its status is a Zep terminal state and reads
    progress.succeeded_items. The facade used to report the internal
    "processed" status with no progress object, so the loop spun until
    its 600s timeout on every real graph build (Oct 27 E2E tracks stuck
    at "0/11 complete"). Driving the real wait loop must return at once
    with the stored episodes' UUIDs."""
    from app import config as config_module
    from app.services.graph_builder import BatchSubmission, GraphBuilderService
    from zep_cloud import BatchAddItem

    monkeypatch.setenv("MIROFISH_MEMORY_BACKEND", "reify")
    monkeypatch.setenv("MIROFISH_REIFY_ROOT", str(tmp_path / "graphs"))
    monkeypatch.setattr(config_module.Config, "ZEP_API_KEY", "")
    clear_zep_client_cache()
    builder = GraphBuilderService(api_key="")
    try:
        builder.client.graph.create(graph_id="g1")
        batch = builder.client.batch.create(
            metadata={"mirofish_operation_id": "op-wait", "graph_id": "g1", "chunk_count": 2}
        )
        items = [
            BatchAddItem(
                type="graph_episode", graph_id="g1", data="Maya runs a solo crew in Charlotte.",
                data_type="text", source_description="chunk",
                metadata={"mirofish_operation_id": "op-wait", "chunk_index": 0},
            ),
            BatchAddItem(
                type="graph_episode", graph_id="g1", data="Darnell worries about shared ZIP codes.",
                data_type="text", source_description="chunk",
                metadata={"mirofish_operation_id": "op-wait", "chunk_index": 1},
            ),
        ]
        added = builder.client.batch.add(batch_id=batch.batch_id, items=items)
        builder.client.batch.process(batch_id=batch.batch_id)
        uuids = builder._wait_for_batch(
            BatchSubmission(
                batch_id=batch.batch_id, operation_id="op-wait",
                episode_uuids=[d.episode_uuid for d in added], item_count=2,
            )
        )
        assert uuids == [d.episode_uuid for d in added]
    finally:
        builder.client.close()


def test_services_construct_without_key_under_reify(monkeypatch, tmp_path):
    from app import config as config_module
    from app.services.zep_entity_reader import ZepEntityReader
    from app.services.zep_tools import ZepToolsService

    monkeypatch.setenv("MIROFISH_MEMORY_BACKEND", "reify")
    monkeypatch.setenv("MIROFISH_REIFY_ROOT", str(tmp_path / "graphs"))
    monkeypatch.setattr(config_module.Config, "ZEP_API_KEY", "")
    reader = ZepEntityReader(api_key=None)
    assert isinstance(reader.client, ReifyZepClient)
    tools = ZepToolsService(api_key=None)
    assert isinstance(tools.client, ReifyZepClient)
    reader.client.close()
    tools.client.close()


def test_default_cloud_path_still_requires_key(monkeypatch):
    from app import config as config_module

    monkeypatch.delenv("MIROFISH_MEMORY_BACKEND", raising=False)
    monkeypatch.setattr(config_module.Config, "ZEP_API_KEY", "")
    clear_zep_client_cache()
    with pytest.raises(ValueError):
        get_zep_client(api_key=None)


def test_default_cloud_path_builds_zep_client(monkeypatch):
    from zep_cloud.client import Zep

    # httpx crashes on bracketed IPv6 literals in no_proxy (known VM
    # egress quirk); sanitize before constructing the Cloud client.
    monkeypatch.setenv("no_proxy", "localhost,127.0.0.1,::1")
    monkeypatch.setenv("NO_PROXY", "localhost,127.0.0.1,::1")
    monkeypatch.delenv("MIROFISH_MEMORY_BACKEND", raising=False)
    clear_zep_client_cache()
    built = get_zep_client(api_key="dummy-key-for-construction-only")
    assert isinstance(built, Zep)
    clear_zep_client_cache()
