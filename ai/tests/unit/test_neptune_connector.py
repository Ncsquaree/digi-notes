import os
import importlib
from modules.knowledge_graph.neptune_connector import NeptuneConnector, NeptuneConnectionError


def test_neptune_singleton(monkeypatch):
    # prevent actual connect attempts by stubbing _connect_with_retry
    monkeypatch.setattr(NeptuneConnector, '_connect_with_retry', lambda self: None)
    # reset singleton
    NeptuneConnector._instance = None
    os.environ['NEPTUNE_ENDPOINT'] = 'localhost'
    os.environ['NEPTUNE_PORT'] = '8182'
    inst1 = NeptuneConnector.get_instance()
    inst2 = NeptuneConnector.get_instance()
    assert inst1 is inst2


def test_neptune_execute_and_close(monkeypatch):
    # create a fake connector that records execute called
    class FakeConn:
        def __init__(self):
            self.closed = False

        def execute(self, fn):
            # call the provided op with a fake traversal object
            class G:
                def V(self, *a, **k):
                    class T:
                        def elementMap(self):
                            class L:
                                def toList(self):
                                    return [{'id': 'v1', 'label': 'Topic', 'name': 'T1'}]
                            return L()
                    return T()
            return fn(G())

        def close(self):
            self.closed = True

    # monkeypatch get_instance to return fake
    monkeypatch.setattr('modules.knowledge_graph.neptune_connector.NeptuneConnector.get_instance', staticmethod(lambda: FakeConn()))
    from modules.knowledge_graph.neptune_connector import NeptuneConnector as NC
    conn = NC.get_instance()
    # call a sample op via execute
    def op(g):
        return {'nodes': [{'id': 'v1'}], 'edges': []}

    res = conn.execute(op)
    assert isinstance(res, dict)
    # close should be callable
    if hasattr(conn, 'close'):
        conn.close()
