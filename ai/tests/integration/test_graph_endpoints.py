import pytest
from fastapi.testclient import TestClient
import main as ai_main


@pytest.mark.integration
def test_graph_visualize_and_related(monkeypatch):
    client = TestClient(ai_main.app)
    # monkeypatch GraphQueries to avoid gremlin dependency
    import modules.knowledge_graph.graph_queries as gq
    monkeypatch.setattr(gq, '__', object())
    class FakeGQ:
        @staticmethod
        def visualize_user_graph(user_id, depth=2):
            return {'nodes': [], 'edges': []}

        @staticmethod
        def get_related_concepts(concept_id, limit=10):
            return []

    monkeypatch.setattr('modules.knowledge_graph.graph_queries.GraphQueries', FakeGQ)
    r = client.get('/graph/visualize/user123')
    assert r.status_code == 200
    rr = client.get('/graph/related-concepts/c1')
    assert rr.status_code == 200
