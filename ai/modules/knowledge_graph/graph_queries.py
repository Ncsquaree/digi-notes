import os
import time
from typing import List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .neptune_connector import NeptuneConnector, NeptuneQueryError
from modules.utils import get_logger

LOG = get_logger()

# Import Gremlin traversal helper `__` with graceful fallback
try:
    try:
        from gremlin_python.process.graph_traversal import __
    except Exception:
        from gremlin_python.process.traversal import __
except Exception:
    __ = None


class GraphQueryError(Exception):
    pass


class GraphQueries:
    @staticmethod
    @retry(retry=retry_if_exception_type(GraphQueryError), reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, max=10))
    def visualize_user_graph(user_id: str, depth: int = 2) -> Dict[str, Any]:
        if __ is None:
            raise GraphQueryError('gremlin traversal helper __ is not available; gremlin_python not installed')
        conn = NeptuneConnector.get_instance()
        max_nodes = int(os.getenv('GRAPH_VISUALIZE_MAX_NODES', '500'))
        max_edges = int(os.getenv('GRAPH_VISUALIZE_MAX_EDGES', '1000'))

        def op(g):
            # start from user's vertices and traverse out up to depth hops
            q = g.V().has('user_id', user_id).repeat(__.outE().inV()).times(depth).emit().dedup().limit(max_nodes)
            # collect nodes as element maps
            nodes = []
            seen = []
            for elem in q.elementMap().toList():
                try:
                    vid = str(elem.get('id'))
                except Exception:
                    vid = str(elem)
                seen.append(vid)
                nodes.append({'id': vid, 'label': elem.get('label'), 'properties': {k: v for k, v in elem.items() if k not in ('id', 'label')}})

            # collect edges incident to seen vertices
            edges = []
            if seen:
                # use spread args to V
                # build traversal to fetch edges and their endpoints
                e_q = g.V(*seen).bothE().limit(max_edges).project('id', 'label', 'out', 'in').by(__.id()).by(__.label()).by(__.outV().id()).by(__.inV().id())
                for e in e_q.toList():
                    edges.append({'id': str(e.get('id')), 'label': e.get('label'), 'source': str(e.get('out')), 'target': str(e.get('in'))})

            return {'nodes': nodes, 'edges': edges}

        try:
            start = time.time()
            res = conn.execute(op)
            elapsed = int((time.time() - start) * 1000)
            LOG.info('visualize_user_graph', extra={'user_id': user_id, 'depth': depth, 'nodes': len(res.get('nodes', [])), 'edges': len(res.get('edges', [])), 'duration_ms': elapsed})
            return res
        except NeptuneQueryError as e:
            LOG.exception('visualize_query_failed', exc_info=True)
            raise GraphQueryError(str(e))

    @staticmethod
    @retry(retry=retry_if_exception_type(GraphQueryError), reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, max=10))
    def get_related_concepts(concept_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        if __ is None:
            raise GraphQueryError('gremlin traversal helper __ is not available; gremlin_python not installed')
        conn = NeptuneConnector.get_instance()

        def op(g):
            # Compute neighbor counts via RELATES_TO edges
            counts_map = {}
            try:
                counts_map = g.V(concept_id).bothE('RELATES_TO').otherV().groupCount().by(__.id()).next()
            except Exception:
                counts_map = {}

            neighbor_ids = list(counts_map.keys())[:limit]
            results = []
            if neighbor_ids:
                # fetch element maps for neighbors
                elems = g.V(*neighbor_ids).elementMap().toList()
                for elem in elems:
                    vid = str(elem.get('id'))
                    props = elem.copy()
                    props.pop('id', None)
                    props.pop('label', None)
                    name = props.get('name') or (props.get('node_label') if props.get('node_label') else None)
                    definition = props.get('definition')
                    strength = int(counts_map.get(elem.get('id'), 0))
                    results.append({'id': vid, 'name': name, 'definition': definition, 'relationship_strength': strength})

            return results

        try:
            start = time.time()
            res = conn.execute(op)
            elapsed = int((time.time() - start) * 1000)
            LOG.info('related_concepts_query', extra={'concept_id': concept_id, 'result_count': len(res), 'duration_ms': elapsed})
            return res
        except NeptuneQueryError as e:
            LOG.exception('related_concepts_failed', exc_info=True)
            raise GraphQueryError(str(e))

    @staticmethod
    @retry(retry=retry_if_exception_type(GraphQueryError), reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, max=10))
    def find_learning_path(start_concept_id: str, end_concept_id: str) -> Dict[str, Any]:
        if __ is None:
            raise GraphQueryError('gremlin traversal helper __ is not available; gremlin_python not installed')
        conn = NeptuneConnector.get_instance()

        def op(g):
            # find shortest path along PREREQUISITE edges
            paths = g.V(start_concept_id).repeat(__.out('PREREQUISITE')).until(__.hasId(end_concept_id)).path().limit(1).toList()
            if not paths:
                return {'error': 'No path found'}
            path = paths[0]
            seq = []
            # path objects may contain vertices; attempt to extract ids
            for elem in getattr(path, 'objects', []) or path:
                try:
                    seq.append(str(getattr(elem, 'id', str(elem))))
                except Exception:
                    seq.append(str(elem))
            return {'path': seq, 'length': len(seq)}

        try:
            res = conn.execute(op)
            return res
        except NeptuneQueryError as e:
            LOG.exception('find_learning_path_failed', exc_info=True)
            raise GraphQueryError(str(e))

    @staticmethod
    def get_concept_prerequisites(concept_id: str) -> List[Dict[str, Any]]:
        if __ is None:
            raise GraphQueryError('gremlin traversal helper __ is not available; gremlin_python not installed')
        conn = NeptuneConnector.get_instance()

        def op(g):
            q = g.V(concept_id).in_('PREREQUISITE')
            res = []
            for v in q.elementMap().toList():
                vid = str(v.get('id'))
                props = {k: v.get(k) for k in v.keys() if k not in ('id', 'label')}
                res.append({'id': vid, 'properties': props})
            return res

        try:
            return conn.execute(op)
        except NeptuneQueryError as e:
            LOG.exception('get_concept_prerequisites_failed', exc_info=True)
            raise GraphQueryError(str(e))

    @staticmethod
    def get_formulas_for_topic(topic_id: str) -> List[Dict[str, Any]]:
        if __ is None:
            raise GraphQueryError('gremlin traversal helper __ is not available; gremlin_python not installed')
        conn = NeptuneConnector.get_instance()

        def op(g):
            # find formulas used in topic or its subtopics
            q = g.V(topic_id).both('CONTAINS').fold().coalesce(__.unfold(), __.V(topic_id)).as_('t').in_('USED_IN')
            res = []
            for v in q.elementMap().toList():
                vid = str(v.get('id'))
                props = {k: v.get(k) for k in v.keys() if k not in ('id', 'label')}
                res.append({'id': vid, 'properties': props})
            return res

        try:
            return conn.execute(op)
        except NeptuneQueryError as e:
            LOG.exception('get_formulas_for_topic_failed', exc_info=True)
            raise GraphQueryError(str(e))
