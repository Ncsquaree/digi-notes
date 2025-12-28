import os
import sqlite3
import tempfile

from knowledge_graph.build import GraphBuilder


class MockEmbGen:
    def __init__(self):
        self.embedding_dim = 8
        self._store = {}
        self._next_id = 1

    def embed_text(self, text):
        # Deterministic small vector based on hash
        h = abs(hash(text))
        return [(h % (i + 7)) / 10.0 for i in range(self.embedding_dim)]

    def store_embedding(self, key, label, vec, entity_type="concept"):
        eid = self._next_id
        self._next_id += 1
        self._store[eid] = vec
        return eid

    def get_embedding_by_id(self, emb_id):
        return self._store.get(emb_id)


def _make_temp_db():
    fd, path = tempfile.mkstemp(suffix=".db", prefix="kg_test_")
    os.close(fd)
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE IF NOT EXISTS flashcards (id INTEGER PRIMARY KEY AUTOINCREMENT, question TEXT, answer TEXT, difficulty INTEGER, context TEXT)"
    )
    cur.execute(
        "INSERT INTO flashcards (question, answer, difficulty, context) VALUES (?,?,?,?)",
        ("What is Photosynthesis?", "Process in plants", 3, "Plants perform photosynthesis in chloroplasts."),
    )
    conn.commit()
    conn.close()
    return path


def test_create_concept_node():
    db_path = _make_temp_db()
    chunks = [
        {
            "text": "Photosynthesis occurs in chloroplasts. It is essential for plants.",
            "metadata": {"type": "paragraph", "topic": "Biology", "entities": [{"text": "Photosynthesis", "type": "concept"}]},
        }
    ]
    emb = MockEmbGen()
    gb = GraphBuilder(chunks_with_entities=chunks, embedding_generator=emb, db_path=db_path)
    res = gb.build_graph()
    assert res["node_type_counts"][GraphBuilder.CONCEPT] >= 1
    assert res["nodes_created"] >= 1


def test_infer_relationships_related_to_edges():
    db_path = _make_temp_db()
    chunks = [
        {
            "text": "Photosynthesis requires light. Chloroplast is an organelle.",
            "metadata": {
                "type": "paragraph",
                "topic": "Biology",
                "entities": [
                    {"text": "Photosynthesis", "type": "concept"},
                    {"text": "Chloroplast", "type": "concept"},
                ],
            },
        },
        {
            "text": "Chloroplast facilitates Photosynthesis in plants.",
            "metadata": {
                "type": "paragraph",
                "topic": "Biology",
                "entities": [
                    {"text": "Chloroplast", "type": "concept"},
                    {"text": "Photosynthesis", "type": "concept"},
                ],
            },
        },
    ]
    emb = MockEmbGen()
    gb = GraphBuilder(chunks_with_entities=chunks, embedding_generator=emb, db_path=db_path)
    res = gb.build_graph()
    assert res["edges_created"] >= 1


def test_link_flashcards_to_entities():
    db_path = _make_temp_db()
    # Prepare chunks with concept to be matched from flashcard text
    chunks = [
        {
            "text": "Photosynthesis occurs in chloroplasts.",
            "metadata": {"type": "paragraph", "topic": "Biology", "entities": [{"text": "Photosynthesis", "type": "concept"}]},
        }
    ]
    # Retrieve flashcard id we inserted
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id FROM flashcards LIMIT 1")
    row = cur.fetchone()
    fc_id = int(row[0])
    conn.close()

    emb = MockEmbGen()
    gb = GraphBuilder(chunks_with_entities=chunks, embedding_generator=emb, db_path=db_path, flashcard_ids=[fc_id])
    res = gb.build_graph()
    # Ensure at least one edge was created from flashcard to some node
    assert res["edges_created"] >= 1


def test_topic_hierarchy_query_returns_children():
    db_path = _make_temp_db()
    chunks = [
        {
            "text": "Science",
            "metadata": {"type": "heading", "topic": "Science", "heading_level": 1},
        },
        {
            "text": "Physics is part of Science.",
            "metadata": {"type": "paragraph", "topic": "Physics"},
        },
    ]
    emb = MockEmbGen()
    gb = GraphBuilder(chunks_with_entities=chunks, embedding_generator=emb, db_path=db_path)
    _ = gb.build_graph()

    from knowledge_graph.query import get_topic_hierarchy
    tree = get_topic_hierarchy(db_path, "Science")
    assert "subtopics" in tree
    labels = [c.get("label") for c in tree["subtopics"]]
    assert "Physics" in labels


def test_formula_linking_non_self_edges():
    db_path = _make_temp_db()
    chunks = [
        {
            "text": "E = mc2 relates energy and mass. Energy is a key concept.",
            "metadata": {
                "type": "paragraph",
                "topic": "Physics",
                "has_formula": True,
                "entities": [{"text": "Energy", "type": "concept"}],
            },
        }
    ]
    emb = MockEmbGen()
    gb = GraphBuilder(chunks_with_entities=chunks, embedding_generator=emb, db_path=db_path)
    _ = gb.build_graph()

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # Find EXPLAINS edges and validate they are not self-loops and target is a Formula node
    cur.execute("SELECT source_id, target_id FROM edges WHERE relationship_type='EXPLAINS'")
    rows = cur.fetchall()
    assert len(rows) >= 1
    for src, tgt in rows:
        assert src != tgt
        cur.execute("SELECT node_type FROM nodes WHERE id=?", (tgt,))
        tgt_type = cur.fetchone()[0]
        assert tgt_type == "Formula"
    conn.close()
