import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from tool_selector.api import create_app
from tool_selector.catalog import Tool, catalog_version, load_catalog
from tool_selector.evaluate import EvalCase, metrics
from tool_selector.search import index_fingerprint


@pytest.fixture
def tools():
    return [Tool(tool_id="users.find", name="find_user", description="Find a user by email",
                 input_schema={"type": "object", "properties": {"email": {"type": "string"}}, "required": ["email"]},
                 retrieval_text="Find a user by email"),
            Tool(tool_id="orders.list", name="list_orders", description="List customer orders",
                 input_schema={"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]},
                 retrieval_text="List orders for a customer")]


class FakeSelector:
    def __init__(self, tools):
        self.tools = tools
        self.version = catalog_version(tools)
        self.last_call = None

    def select(self, query, top_k):
        self.last_call = (query, top_k)
        return self.tools[:top_k]


def test_api_returns_full_schemas_and_hides_search_text(tools):
    engine = FakeSelector(tools)
    with TestClient(create_app(engine)) as client:
        response = client.post("/select-tools", json={"query": "  Find a user  ", "top_k": 1})
    assert response.status_code == 200
    body = response.json()
    assert body["catalog_version"] == engine.version
    assert len(body["tools"]) == 1
    assert body["tools"][0]["input_schema"]["required"] == ["email"]
    assert "retrieval_text" not in body["tools"][0]
    assert engine.last_call == ("Find a user", 1)


def test_api_default_k_is_15(tools):
    engine = FakeSelector(tools)
    with TestClient(create_app(engine)) as client:
        assert client.post("/select-tools", json={"query": "Find user"}).status_code == 200
    assert engine.last_call == ("Find user", 15)


@pytest.mark.parametrize("body", [{}, {"query": " "}, {"query": "x", "top_k": 0},
                                 {"query": "x", "top_k": 101}, {"query": "x", "extra": 1}])
def test_api_validation(tools, body):
    with TestClient(create_app(FakeSelector(tools))) as client:
        assert client.post("/select-tools", json=body).status_code == 422


def test_metrics_measure_partial_and_complete_coverage():
    assert metrics(["wrong", "a", "b"], ["a", "b"], 2) == {"recall": 0.5, "complete": 0.0}
    assert metrics(["wrong", "a", "b"], ["a", "b"], 3) == {"recall": 1.0, "complete": 1.0}
    assert metrics([], ["a"], 5)["recall"] == 0
    with pytest.raises(ValidationError):
        EvalCase(query_id="q", query="Find user", relevant_tool_ids=[])


def test_duplicate_catalog_ids_rejected(tmp_path, tools):
    row = {**tools[0].model_dump(), "retrieval_text": tools[0].retrieval_text}
    path = tmp_path / "tools.jsonl"
    path.write_text(json.dumps(row) + "\n" + json.dumps(row))
    with pytest.raises(ValueError, match="Duplicate tool_id"):
        load_catalog(path)


def test_index_invalidates_when_search_text_or_order_changes(tools):
    changed = tools[0].model_copy(update={"retrieval_text": "A different operation"})
    assert index_fingerprint([changed, tools[1]]) != index_fingerprint(tools)
    assert index_fingerprint(list(reversed(tools))) != index_fingerprint(tools)
    assert catalog_version([changed, tools[1]]) != catalog_version(tools)


def test_api_rejects_cyrillic_with_language_explanation(tools):
    with TestClient(create_app(FakeSelector(tools))) as client:
        response = client.post("/select-tools", json={"query": "Найди отели поблизости"})
    assert response.status_code == 422
    assert "English" in response.text
