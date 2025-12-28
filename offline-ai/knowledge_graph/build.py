"""
SQLite-based Knowledge Graph Builder (Phase 6)

Implements offline graph construction mirroring the Neptune-based system.
Creates nodes (Concept, Topic, Entity, Flashcard) and edges (RELATED_TO, PART_OF,
EXPLAINS, DERIVED_FROM), stores embeddings for nodes, and links flashcards to entities.
"""

import sqlite3
import json
import re
from typing import List, Dict, Any, Optional

import numpy as np


class GraphBuilder:
  CONCEPT = "Concept"
  TOPIC = "Topic"
  ENTITY = "Entity"
  FLASHCARD = "Flashcard"
  FORMULA = "Formula"

  RELATED_TO = "RELATED_TO"
  PART_OF = "PART_OF"
  EXPLAINS = "EXPLAINS"
  DERIVED_FROM = "DERIVED_FROM"

  def __init__(
    self,
    chunks_with_entities: List[Dict[str, Any]],
    embedding_generator: Any,
    db_path: str,
    flashcard_ids: Optional[List[int]] = None,
  ) -> None:
    self.chunks = chunks_with_entities or []
    self.emb = embedding_generator
    self.db_path = db_path
    self.flashcard_ids = flashcard_ids or []

    self.nodes_created = 0
    self.edges_created = 0
    self.node_id_map: Dict[str, int] = {}
    self.node_type_counts = {self.CONCEPT: 0, self.TOPIC: 0, self.ENTITY: 0, self.FLASHCARD: 0, self.FORMULA: 0}

    self._conn = sqlite3.connect(self.db_path)
    self._conn.row_factory = sqlite3.Row
    self._cur = self._conn.cursor()
    self._ensure_schema()

  def _ensure_schema(self) -> None:
    # Ensure nodes and edges tables exist (no-op if already present)
    self._cur.execute(
      """
      CREATE TABLE IF NOT EXISTS nodes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        node_type TEXT NOT NULL,
        label TEXT NOT NULL,
        properties TEXT,
        embedding_id INTEGER
      )
      """
    )
    self._cur.execute(
      """
      CREATE TABLE IF NOT EXISTS edges (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_id INTEGER NOT NULL,
        target_id INTEGER NOT NULL,
        relationship_type TEXT NOT NULL,
        weight REAL,
        FOREIGN KEY(source_id) REFERENCES nodes(id),
        FOREIGN KEY(target_id) REFERENCES nodes(id)
      )
      """
    )
    self._conn.commit()

  def build_graph(self) -> Dict[str, Any]:
    concepts = self._extract_concepts(self.chunks)
    topics = self._extract_topics(self.chunks)
    entities = self._extract_entities(self.chunks)

    concept_nodes = {}
    for concept_label, chunk_text in concepts.items():
      if concept_label in self.node_id_map:
        continue
      node_id = self._create_concept_node({"name": concept_label}, chunk_text)
      concept_nodes[concept_label] = node_id

    topic_nodes = {}
    for topic_label, topic_chunks in topics.items():
      if topic_label in self.node_id_map:
        continue
      node_id = self._create_topic_node(topic_label, topic_chunks)
      topic_nodes[topic_label] = node_id

    entity_nodes = {}
    for ent in entities:
      key = ent.get("text") or ent.get("name")
      if not key or key in self.node_id_map:
        continue
      node_id = self._create_entity_node(ent)
      entity_nodes[key] = node_id

    self._infer_concept_relationships(concept_nodes)
    self._infer_topic_hierarchy(topic_nodes, self.chunks)
    self._link_formulas_to_concepts(self.chunks)

    # Link flashcards to detected entities/concepts/topics
    for fc_id in self.flashcard_ids:
      try:
        self._link_flashcard_node(fc_id)
      except Exception:
        continue

    result = {
      "nodes_created": self.nodes_created,
      "edges_created": self.edges_created,
      "node_type_counts": self.node_type_counts,
      "node_ids": self.node_id_map,
    }
    return result

  # --- Extraction helpers ---
  def _extract_concepts(self, chunks: List[Dict[str, Any]]) -> Dict[str, str]:
    concepts: Dict[str, str] = {}
    for ch in chunks:
      text = ch.get("text", "")
      # entities may be under metadata or top-level; support both
      ents = ch.get("entities") or ch.get("metadata", {}).get("entities", [])
      for ent in ents:
        etype = (ent.get("type") or "").lower()
        if etype in {"concept", "term", "topic"}:
          name = ent.get("text") or ent.get("name")
          if not name:
            continue
          concepts.setdefault(name, text)
    return concepts

  def _extract_topics(self, chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    topics: Dict[str, List[Dict[str, Any]]] = {}
    for ch in chunks:
      topic = ch.get("topic") or ch.get("metadata", {}).get("topic") or ch.get("heading")
      if topic:
        topics.setdefault(topic, []).append(ch)
    return topics

  def _extract_entities(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    uniq: Dict[str, Dict[str, Any]] = {}
    for ch in chunks:
      ents = ch.get("entities") or ch.get("metadata", {}).get("entities", [])
      for ent in ents:
        name = ent.get("text") or ent.get("name")
        if not name:
          continue
        t = (ent.get("type") or "").lower()
        if t in {"person", "org", "gpe", "entity", "proper_noun", "formula"}:
          uniq[name] = ent
    return list(uniq.values())

  # --- Node creation methods ---
  def _create_concept_node(self, entity_dict: Dict[str, Any], chunk_text: str) -> int:
    name = entity_dict.get("name") or entity_dict.get("text") or ""
    definition = self._first_sentence_mentioning(chunk_text, name)
    examples = self._extract_examples(chunk_text)

    emb_vec = self._safe_embed_text(name)
    emb_id = self._safe_store_embedding(f"concept-{name}", name, emb_vec, entity_type="concept")

    props = json.dumps({"definition": definition, "examples": examples})
    node_id = self._insert_node(self.CONCEPT, name, props, emb_id)
    self.node_id_map[name] = node_id
    self.node_type_counts[self.CONCEPT] += 1
    self.nodes_created += 1
    return node_id

  def _create_topic_node(self, topic_name: str, chunks: List[Dict[str, Any]]) -> int:
    description = self._first_paragraph(chunks)
    key_points = self._extract_key_points(chunks)

    emb_vec = self._safe_embed_text(topic_name)
    emb_id = self._safe_store_embedding(f"topic-{topic_name}", topic_name, emb_vec, entity_type="topic")

    props = json.dumps({"description": description, "key_points": key_points})
    node_id = self._insert_node(self.TOPIC, topic_name, props, emb_id)
    self.node_id_map[topic_name] = node_id
    self.node_type_counts[self.TOPIC] += 1
    self.nodes_created += 1
    return node_id

  def _create_entity_node(self, entity_dict: Dict[str, Any]) -> int:
    name = entity_dict.get("text") or entity_dict.get("name") or ""
    entity_type = entity_dict.get("type") or "entity"
    context = entity_dict.get("context") or entity_dict.get("source_text") or ""

    emb_vec = self._safe_embed_text(name)
    emb_id = self._safe_store_embedding(f"entity-{name}", name, emb_vec, entity_type=entity_type)

    props = json.dumps({"entity_type": entity_type, "context": context})
    node_id = self._insert_node(self.ENTITY, name, props, emb_id)
    self.node_id_map[name] = node_id
    self.node_type_counts[self.ENTITY] += 1
    self.nodes_created += 1
    return node_id

  # --- Edge creation and inference ---
  def _infer_concept_relationships(self, concept_nodes: Dict[str, int]) -> None:
    labels = list(concept_nodes.keys())
    for i in range(len(labels)):
      for j in range(i + 1, len(labels)):
        a, b = labels[i], labels[j]
        a_id, b_id = concept_nodes[a], concept_nodes[b]

        a_def = self._get_property(a_id, "definition")
        b_def = self._get_property(b_id, "definition")
        if a_def and b.lower() in a_def.lower():
          self._insert_edge(a_id, b_id, self.RELATED_TO, 0.8)
        if b_def and a.lower() in b_def.lower():
          self._insert_edge(b_id, a_id, self.RELATED_TO, 0.8)

    emb_map: Dict[int, Optional[np.ndarray]] = {}
    for label, node_id in concept_nodes.items():
      emb_map[node_id] = self._load_embedding_vector(node_id)

    def cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
      if a is None or b is None:
        return 0.0
      denom = (np.linalg.norm(a) * np.linalg.norm(b))
      if denom == 0:
        return 0.0
      return float(np.dot(a, b) / denom)

    for i in range(len(labels)):
      for j in range(i + 1, len(labels)):
        a_id = concept_nodes[labels[i]]
        b_id = concept_nodes[labels[j]]
        sim = cosine(emb_map[a_id], emb_map[b_id])
        if sim > 0.7:
          self._insert_edge(a_id, b_id, self.RELATED_TO, sim)
          self._insert_edge(b_id, a_id, self.RELATED_TO, sim)

  def _infer_topic_hierarchy(self, topic_nodes: Dict[str, int], chunks: List[Dict[str, Any]]) -> None:
    ordered = sorted([(ch.get("metadata", {}).get("heading_level") or 0, ch) for ch in chunks], key=lambda x: x[0])
    for idx, (_, ch) in enumerate(ordered):
      if (ch.get("metadata", {}).get("type") or ch.get("chunk_type") or "").lower() == "heading":
        parent_topic = ch.get("metadata", {}).get("topic") or ch.get("topic") or ch.get("heading")
        if not parent_topic or parent_topic not in topic_nodes:
          continue
        if idx + 1 < len(ordered):
          next_ch = ordered[idx + 1][1]
          child_topic = next_ch.get("metadata", {}).get("topic") or next_ch.get("topic") or next_ch.get("heading")
          if child_topic and child_topic in topic_nodes:
            self._insert_edge(topic_nodes[child_topic], topic_nodes[parent_topic], self.PART_OF, 1.0)

    for a_label, a_id in topic_nodes.items():
      a_desc = self._get_property(a_id, "description") or ""
      for b_label, b_id in topic_nodes.items():
        if a_label == b_label:
          continue
        if b_label.lower() in a_desc.lower():
          self._insert_edge(b_id, a_id, self.PART_OF, 0.9)

  def _link_formulas_to_concepts(self, chunks: List[Dict[str, Any]]) -> None:
    formula_re = re.compile(r"[A-Za-z0-9]+\s*[=+×÷→⇒]\s*[A-Za-z0-9+×÷→⇒\s]+")
    for ch in chunks:
      if not (ch.get("metadata", {}).get("has_formula") or ch.get("has_formula")):
        continue
      text = ch.get("text", "")
      formulas = formula_re.findall(text)
      if not formulas:
        continue
      # Create/reuse formula nodes
      for formula in formulas:
        formula_label = f"formula:{formula.strip()}"
        if formula_label in self.node_id_map:
          formula_node_id = self.node_id_map[formula_label]
        else:
          emb_vec = self._safe_embed_text(formula)
          emb_id = self._safe_store_embedding(f"formula-{formula}", formula, emb_vec, entity_type="formula")
          props = json.dumps({"text": formula})
          formula_node_id = self._insert_node(self.FORMULA, formula_label, props, emb_id)
          self.node_id_map[formula_label] = formula_node_id
          self.node_type_counts[self.FORMULA] += 1
          self.nodes_created += 1
      # Link concepts/topics in the same chunk to the formula nodes
      for formula in formulas:
        formula_label = f"formula:{formula.strip()}"
        formula_node_id = self.node_id_map.get(formula_label)
        if not formula_node_id:
          continue
        for label, node_id in list(self.node_id_map.items()):
          if label.startswith("flashcard-"):
            continue
          ntype = self._get_node_type(node_id)
          if ntype in {self.CONCEPT, self.TOPIC}:
            if label.lower() in (text or "").lower():
              if node_id != formula_node_id:
                self._insert_edge(node_id, formula_node_id, self.EXPLAINS, 0.6)

  # --- Flashcard linking ---
  def _link_flashcard_node(self, flashcard_id: int) -> None:
    self._cur.execute(
      "SELECT id, question, answer, context FROM flashcards WHERE id=?",
      (flashcard_id,),
    )
    row = self._cur.fetchone()
    if not row:
      return
    question = (row[1] or "").lower()
    answer = (row[2] or "").lower()
    context = (row[3] or "").lower()

    fc_label = f"flashcard-{flashcard_id}"
    if fc_label in self.node_id_map:
      fc_node_id = self.node_id_map[fc_label]
    else:
      props = json.dumps({"question": row[1], "answer": row[2], "context": row[3]})
      fc_node_id = self._insert_node(self.FLASHCARD, fc_label, props, None)
      self.node_id_map[fc_label] = fc_node_id
      self.node_type_counts[self.FLASHCARD] += 1
      self.nodes_created += 1

    linked = 0
    for label, node_id in list(self.node_id_map.items()):
      if label.startswith("flashcard-"):
        continue
      l = label.lower()
      if l in question or l in answer or l in context:
        self._insert_edge(fc_node_id, node_id, self.DERIVED_FROM, 1.0)
        linked += 1

    if linked == 0:
      for label, node_id in list(self.node_id_map.items()):
        if self._get_node_type(node_id) == self.TOPIC:
          self._insert_edge(fc_node_id, node_id, self.DERIVED_FROM, 0.5)
          break

  # --- DB helpers ---
  def _insert_node(self, node_type: str, label: str, properties_json: str, embedding_id: Optional[int]) -> int:
    self._cur.execute(
      "INSERT INTO nodes (node_type, label, properties, embedding_id) VALUES (?, ?, ?, ?)",
      (node_type, label, properties_json, embedding_id),
    )
    self._conn.commit()
    return int(self._cur.lastrowid)

  def _insert_edge(self, source_id: int, target_id: int, relationship_type: str, weight: float) -> None:
    self._cur.execute(
      "INSERT INTO edges (source_id, target_id, relationship_type, weight) VALUES (?, ?, ?, ?)",
      (source_id, target_id, relationship_type, float(weight)),
    )
    self._conn.commit()
    self.edges_created += 1

  def _get_property(self, node_id: int, key: str) -> Optional[str]:
    self._cur.execute("SELECT properties FROM nodes WHERE id=?", (node_id,))
    row = self._cur.fetchone()
    if not row or not row[0]:
      return None
    try:
      props = json.loads(row[0])
      val = props.get(key)
      if isinstance(val, (str, int, float)):
        return str(val)
      return json.dumps(val)
    except Exception:
      return None

  def _get_node_type(self, node_id: int) -> Optional[str]:
    self._cur.execute("SELECT node_type FROM nodes WHERE id=?", (node_id,))
    r = self._cur.fetchone()
    return r[0] if r else None

  def _load_embedding_vector(self, node_id: int) -> Optional[np.ndarray]:
    try:
      self._cur.execute("SELECT embedding_id FROM nodes WHERE id=?", (node_id,))
      r = self._cur.fetchone()
      if not r or r[0] is None:
        return None
      emb_id = int(r[0])
      if hasattr(self.emb, "get_embedding_by_id"):
        vec = self.emb.get_embedding_by_id(emb_id)
        if vec is None:
          return None
        return np.array(vec, dtype=np.float32)
      return None
    except Exception:
      return None

  # --- Text helpers ---
  def _first_sentence_mentioning(self, text: str, name: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text or "")
    name_l = (name or "").lower()
    for s in sentences:
      if name_l and name_l in s.lower():
        return s.strip()
    return (sentences[0] if sentences else "").strip()

  def _first_paragraph(self, chunks: List[Dict[str, Any]]) -> str:
    for ch in chunks:
      if (ch.get("chunk_type") or ch.get("metadata", {}).get("type") or "").lower() in {"paragraph", "text"}:
        return (ch.get("text") or "").strip()
    return (chunks[0].get("text") or "").strip() if chunks else ""

  def _extract_key_points(self, chunks: List[Dict[str, Any]]) -> List[str]:
    points: List[str] = []
    for ch in chunks:
      if (ch.get("chunk_type") or ch.get("metadata", {}).get("type") or "").lower() == "bullet":
        points.append((ch.get("text") or "").strip())
    return points[:10]

  def _extract_examples(self, text: str) -> List[str]:
    examples: List[str] = []
    for line in (text or "").splitlines():
      if line.strip().lower().startswith(("e.g.", "example", "for instance")):
        examples.append(line.strip())
    return examples[:5]

  # --- Embedding helpers ---
  def _safe_embed_text(self, text: str) -> Optional[List[float]]:
    try:
      if hasattr(self.emb, "embed_text"):
        return self.emb.embed_text(text)
    except Exception:
      return None
    return None

  def _safe_store_embedding(self, key: str, label: str, vec: Optional[List[float]], entity_type: str) -> Optional[int]:
    try:
      if hasattr(self.emb, "store_embedding"):
        return self.emb.store_embedding(key, label, vec, entity_type=entity_type)
    except Exception:
      return None
    return None

