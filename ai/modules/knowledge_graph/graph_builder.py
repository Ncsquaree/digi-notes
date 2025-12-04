import os
import json
from typing import List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from modules.semantic import ParsedContent
from .neptune_connector import NeptuneConnector, NeptuneQueryError
from modules.utils import get_logger

LOG = get_logger()


class GraphBuildError(Exception):
    pass


class GraphBuilder:
    TOPIC = 'Topic'
    SUBTOPIC = 'Subtopic'
    CONCEPT = 'Concept'
    FORMULA = 'Formula'

    CONTAINS = 'CONTAINS'
    RELATES_TO = 'RELATES_TO'
    PREREQUISITE = 'PREREQUISITE'
    USED_IN = 'USED_IN'

    def __init__(self, user_id: str, note_id: str, parsed_content: ParsedContent):
        self.user_id = user_id
        self.note_id = note_id
        if isinstance(parsed_content, dict):
            self.parsed = ParsedContent.model_validate(parsed_content)
        else:
            self.parsed = parsed_content
        self.conn = NeptuneConnector.get_instance()
        self.vertex_ids: Dict[str, str] = {}
        self.nodes_created = 0
        self.edges_created = 0
        # map of vertex_key -> properties used when creating the vertex
        self._props_map: Dict[str, Dict[str, Any]] = {}

    def build_graph(self) -> Dict[str, Any]:
        try:
            topics = getattr(self.parsed, 'topics', []) or []
            concepts = getattr(self.parsed, 'key_concepts', []) or []
            formulas = getattr(self.parsed, 'formulas', []) or []

            # create topics and subtopics
            for t in topics:
                t_id = self._create_topic_node(t)
                # subtopics
                for s in (t.subtopics or []):
                    s_id = self._create_subtopic_node(s, parent_topic_id=t_id)

            # concepts
            for c in concepts:
                self._create_concept_node(c)

            # formulas
            for f in formulas:
                self._create_formula_node(f)

            # infer relationships
            rels = self._infer_concept_relationships(concepts)
            for src, tgt in rels:
                self._create_concept_edge(src, tgt, self.RELATES_TO)

            used_pairs = self._infer_formula_usage(formulas, topics)
            for fnode, tnode in used_pairs:
                self._create_edge(fnode, tnode, self.USED_IN)

            # simple prerequisite inference based on academic_level
            try:
                self._infer_prerequisites(topics)
            except Exception:
                LOG.warning('prerequisite_infer_failed')

            return {'nodes_created': self.nodes_created, 'edges_created': self.edges_created, 'vertex_ids': self.vertex_ids}
        except NeptuneQueryError as e:
            LOG.exception('graph_build_neptune_error', exc_info=True)
            raise GraphBuildError(str(e))
        except Exception as e:
            LOG.exception('graph_build_error', exc_info=True)
            raise GraphBuildError(str(e))

    def _create_topic_node(self, topic) -> str:
        label = topic.title if getattr(topic, 'title', None) else getattr(topic, 'name', 'topic')
        props = {
            'name': label,
            'description': getattr(topic, 'description', None),
            'academic_level': getattr(topic, 'metadata', {}).get('academic_level') if getattr(topic, 'metadata', None) else None,
        }
        vid = self._add_vertex(self.TOPIC, label, props)
        key = f"topic:{label}"
        self.vertex_ids[key] = vid
        return vid

    def _create_subtopic_node(self, subtopic, parent_topic_id: str) -> str:
        label = getattr(subtopic, 'title', None) or getattr(subtopic, 'name', None) or 'subtopic'
        props = {
            'name': label,
            'content': getattr(subtopic, 'content', None),
            'key_points': getattr(subtopic, 'key_points', None),
        }
        vid = self._add_vertex(self.SUBTOPIC, label, props)
        self.vertex_ids[f"subtopic:{label}"] = vid
        if parent_topic_id:
            self._create_edge_by_id(parent_topic_id, vid, self.CONTAINS)
        return vid

    def _create_concept_node(self, concept) -> str:
        name = getattr(concept, 'name', None) or getattr(concept, 'label', None) or 'concept'
        props = {
            'name': name,
            'definition': getattr(concept, 'definition', None),
            'examples': getattr(concept, 'examples', None),
        }
        vid = self._add_vertex(self.CONCEPT, name, props)
        self.vertex_ids[f"concept:{name}"] = vid
        return vid

    def _create_formula_node(self, formula) -> str:
        latex = getattr(formula, 'latex', None) or getattr(formula, 'expression', None) or 'formula'
        props = {
            'latex': latex,
            'description': getattr(formula, 'description', None),
            'variables': getattr(formula, 'variables', None),
        }
        vid = self._add_vertex(self.FORMULA, latex, props)
        self.vertex_ids[f"formula:{latex}"] = vid
        return vid

    def _infer_concept_relationships(self, concepts: List) -> List[tuple]:
        pairs = []
        names = [getattr(c, 'name', None) or getattr(c, 'label', None) for c in concepts]
        for c in concepts:
            src = getattr(c, 'name', None) or getattr(c, 'label', None)
            text = ' '.join(filter(None, [getattr(c, 'definition', '') or '', ' '.join(getattr(c, 'examples', []) or [])]))
            if not src:
                continue
            for other in names:
                if not other or other.lower() == src.lower():
                    continue
                if other.lower() in (text or '').lower():
                    pairs.append((src, other))
        return pairs

    def _infer_formula_usage(self, formulas: List, topics: List) -> List[tuple]:
        pairs = []
        # naive matching: look for formula latex or description in topic/subtopic content
        for f in formulas:
            fkey = getattr(f, 'latex', None) or getattr(f, 'expression', None) or ''
            fnode = self.vertex_ids.get(f"formula:{fkey}")
            if not fnode:
                continue
            for t in topics:
                tname = getattr(t, 'title', None) or getattr(t, 'name', None)
                tcontent = ' '.join([getattr(t, 'description', '') or ''] + [getattr(s, 'content', '') or '' for s in (t.subtopics or [])])
                if fkey and fkey and (fkey in (tcontent or '') or (getattr(f, 'description', '') or '') in (tcontent or '')):
                    tnode = self.vertex_ids.get(f"topic:{tname}")
                    if tnode:
                        pairs.append((fnode, tnode))
        return pairs

    def _infer_prerequisites(self, topics: List):
        # Concept-level prerequisite inference (preferred):
        # Use concept.metadata.academic_level if present, or fall back to name-based heuristics.
        level_order = {'primary': 0, 'middle': 1, 'high school': 2, 'undergraduate': 3, 'graduate': 4}
        cmap = []
        concepts = getattr(self.parsed, 'key_concepts', []) or []
        for c in concepts:
            name = getattr(c, 'name', None) or getattr(c, 'label', None)
            lvl = None
            if getattr(c, 'metadata', None):
                lvl = c.metadata.get('academic_level')
            cmap.append((name, lvl))

        for a, al in cmap:
            for b, bl in cmap:
                if not a or not b or a == b or not al or not bl:
                    continue
                if level_order.get(al, -1) < level_order.get(bl, -1):
                    # create prerequisite edge between concepts (a -> b)
                    a_vid = self.vertex_ids.get(f"concept:{a}")
                    b_vid = self.vertex_ids.get(f"concept:{b}")
                    if a_vid and b_vid:
                        self._create_edge_by_id(a_vid, b_vid, self.PREREQUISITE)

    def _add_vertex(self, label: str, node_label: str, properties: Dict[str, Any]) -> str:
        key = f"{label}:{node_label}"
        # store props for later Postgres sync
        try:
            self._props_map[key] = properties or {}
        except Exception:
            self._props_map[key] = {}

        def op(g):
            v = g.addV(label).property('user_id', self.user_id).property('note_id', self.note_id).property('node_label', node_label)
            for k, vprop in (properties or {}).items():
                if vprop is None:
                    continue
                # store complex props as JSON string
                if isinstance(vprop, (dict, list)):
                    try:
                        v = v.property(k, json.dumps(vprop))
                    except Exception:
                        v = v.property(k, str(vprop))
                else:
                    v = v.property(k, str(vprop))
            # execute and return id
            res = v.next()
            try:
                vid = getattr(res, 'id', str(res))
            except Exception:
                vid = str(res)
            return str(vid)

        vid = self.conn.execute(op)
        self.nodes_created += 1
        LOG.debug('node_created', extra={'node_type': label, 'node_label': node_label, 'vertex_id': vid})
        return vid

    def _create_edge_by_id(self, src_vid: str, tgt_vid: str, label: str):
        def op(g):
            g.V(src_vid).addE(label).to(g.V(tgt_vid)).next()

        self.conn.execute(op)
        self.edges_created += 1
        LOG.debug('edge_created', extra={'edge_label': label, 'source': src_vid, 'target': tgt_vid})

    def _create_edge(self, src_key: str, tgt_key: str, label: str):
        src = self.vertex_ids.get(src_key)
        tgt = self.vertex_ids.get(tgt_key)
        if src and tgt:
            self._create_edge_by_id(src, tgt, label)

    def _create_concept_edge(self, src_name: str, tgt_name: str, label: str):
        src = self.vertex_ids.get(f"concept:{src_name}")
        tgt = self.vertex_ids.get(f"concept:{tgt_name}")
        if src and tgt:
            self._create_edge_by_id(src, tgt, label)

    def sync_to_postgres(self, vertex_ids: Dict[str, str], db_pool):
        # optional helper to persist vertex ids to Postgres knowledge_graph_nodes table
        try:
            import psycopg2
            conn = db_pool.getconn()
            cur = conn.cursor()
            for key, vid in vertex_ids.items():
                node_type, node_label = key.split(':', 1)
                props = self._props_map.get(key, {})
                try:
                    props_json = json.dumps(props)
                except Exception:
                    props_json = json.dumps({})
                cur.execute(
                    "INSERT INTO knowledge_graph_nodes (node_type, node_label, neptune_vertex_id, properties) VALUES (%s, %s, %s, %s) ON CONFLICT (neptune_vertex_id) DO NOTHING",
                    (node_type, node_label, vid, props_json),
                )
            conn.commit()
            cur.close()
            db_pool.putconn(conn)
        except Exception as e:
            LOG.exception('sync_to_postgres_failed', exc_info=True)
            # don't raise — allow build to succeed even if sync fails
            return
