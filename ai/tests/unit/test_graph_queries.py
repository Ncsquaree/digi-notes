from modules.knowledge_graph import graph_queries


def test_visualize_user_graph(monkeypatch):
    # prepare fake connector
    class FakeConn:
        def execute(self, fn):
            # simulate op returning nodes and edges
            return {'nodes': [{'id': '1', 'label': 'Topic', 'properties': {'name': 'T1'}}], 'edges': [{'id': 'e1', 'label': 'RELATES_TO', 'source': '1', 'target': '2'}]}

    monkeypatch.setattr('modules.knowledge_graph.graph_queries.__', object())
    monkeypatch.setattr('modules.knowledge_graph.graph_queries.NeptuneConnector', type('X', (), {'get_instance': staticmethod(lambda: FakeConn())}))
    res = graph_queries.GraphQueries.visualize_user_graph('user1', depth=2)
    assert 'nodes' in res and isinstance(res['nodes'], list)


def test_get_related_concepts(monkeypatch):
    class FakeConn:
        def execute(self, fn):
            return [{'id': 'c1', 'name': 'Concept1', 'definition': 'Def', 'relationship_strength': 2}]

    monkeypatch.setattr('modules.knowledge_graph.graph_queries.__', object())
    monkeypatch.setattr('modules.knowledge_graph.graph_queries.NeptuneConnector', type('X', (), {'get_instance': staticmethod(lambda: FakeConn())}))
    res = graph_queries.GraphQueries.get_related_concepts('c1', limit=5)
    assert isinstance(res, list)


def test_find_learning_path(monkeypatch):
    class FakeConn:
        def execute(self, fn):
            return {'path': ['a', 'b', 'c'], 'length': 3}

    monkeypatch.setattr('modules.knowledge_graph.graph_queries.__', object())
    monkeypatch.setattr('modules.knowledge_graph.graph_queries.NeptuneConnector', type('X', (), {'get_instance': staticmethod(lambda: FakeConn())}))
    res = graph_queries.GraphQueries.find_learning_path('a', 'c')
    assert isinstance(res, dict)
