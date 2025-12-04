from modules.knowledge_graph.graph_builder import GraphBuilder
from modules.semantic import ParsedContent


class FakeConn:
    def __init__(self):
        self._next = 1

    def execute(self, fn):
        # call callable with a fake traversal object; just return an id
        nid = f'vertex-{self._next}'
        self._next += 1
        return nid


def test_graph_builder_monkeypatch(monkeypatch, sample_parsed_content):
    # monkeypatch NeptuneConnector.get_instance to return a fake connection
    from modules.knowledge_graph import neptune_connector

    class FakeNeptune:
        def __init__(self):
            self._fake = FakeConn()

        def execute(self, fn):
            return self._fake.execute(fn)

    monkeypatch.setattr('modules.knowledge_graph.neptune_connector.NeptuneConnector.get_instance', staticmethod(lambda: FakeNeptune()))

    gb = GraphBuilder(user_id='u1', note_id='n1', parsed_content=sample_parsed_content)
    out = gb.build_graph()
    assert isinstance(out, dict)
    assert 'nodes_created' in out
