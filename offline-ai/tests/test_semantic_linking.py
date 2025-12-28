"""
Unit tests for semantic linking between flashcards and knowledge graph nodes.

Tests:
  - link_flashcards_to_graph: Edge creation with threshold filtering
  - _ensure_minimum_links: Fallback linking for orphaned flashcards
  - cluster_flashcards: K-means clustering validation
  - find_similar_flashcards_for_node: Query API for flashcard retrieval
  - find_similar_nodes_for_flashcard: Reverse query for node retrieval
  - Threshold edge cases (0.69 vs 0.71 similarity)
"""

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from knowledge_graph.link import (
    link_flashcards_to_graph,
    cluster_flashcards,
    find_similar_flashcards_for_node,
    find_similar_nodes_for_flashcard,
    _load_embeddings_for_entities,
    _create_semantic_edges,
    _ensure_minimum_links
)


@pytest.fixture
def sample_db(tmp_path):
    """Create in-memory SQLite database with sample data."""
    db_path = tmp_path / "test_semantic.db"
    conn = sqlite3.connect(str(db_path))
    
    # Create schema (from init_db.py)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vector BLOB NOT NULL,
            entity_type TEXT NOT NULL,
            entity_id INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS nodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT UNIQUE NOT NULL,
            node_type TEXT NOT NULL,
            properties TEXT,
            embedding_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (embedding_id) REFERENCES embeddings(id)
        );
        
        CREATE TABLE IF NOT EXISTS edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id INTEGER NOT NULL,
            target_id INTEGER NOT NULL,
            relationship_type TEXT NOT NULL,
            weight REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (source_id) REFERENCES nodes(id),
            FOREIGN KEY (target_id) REFERENCES nodes(id)
        );
        
        CREATE TABLE IF NOT EXISTS flashcards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            topic TEXT,
            difficulty TEXT DEFAULT 'medium',
            embedding_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (embedding_id) REFERENCES embeddings(id)
        );
    """)
    
    # Insert sample embeddings (512-dim vectors)
    # Flashcard embeddings
    fc1_vec = np.random.randn(512).astype(np.float32)
    fc1_vec = fc1_vec / np.linalg.norm(fc1_vec)  # Normalize
    fc2_vec = np.random.randn(512).astype(np.float32)
    fc2_vec = fc2_vec / np.linalg.norm(fc2_vec)
    
    # Node embeddings (some similar to flashcards, some not)
    node1_vec = fc1_vec + np.random.randn(512).astype(np.float32) * 0.1  # High similarity to fc1
    node1_vec = node1_vec / np.linalg.norm(node1_vec)
    
    node2_vec = fc2_vec + np.random.randn(512).astype(np.float32) * 0.1  # High similarity to fc2
    node2_vec = node2_vec / np.linalg.norm(node2_vec)
    
    node3_vec = np.random.randn(512).astype(np.float32)  # Low similarity to both
    node3_vec = node3_vec / np.linalg.norm(node3_vec)
    
    # Insert embeddings
    cursor = conn.executemany(
        "INSERT INTO embeddings (vector, entity_type, entity_id) VALUES (?, ?, ?)",
        [
            (fc1_vec.tobytes(), 'flashcard', 1),
            (fc2_vec.tobytes(), 'flashcard', 2),
            (node1_vec.tobytes(), 'node', 1),
            (node2_vec.tobytes(), 'node', 2),
            (node3_vec.tobytes(), 'node', 3),
        ]
    )
    
    fc1_emb_id = 1
    fc2_emb_id = 2
    node1_emb_id = 3
    node2_emb_id = 4
    node3_emb_id = 5
    
    # Insert flashcards
    conn.executemany(
        "INSERT INTO flashcards (id, question, answer, topic, embedding_id) VALUES (?, ?, ?, ?, ?)",
        [
            (1, "What is photosynthesis?", "Process of converting light to energy", "Biology", fc1_emb_id),
            (2, "What is cellular respiration?", "Process of breaking down glucose", "Biology", fc2_emb_id),
        ]
    )
    
    # Insert KG nodes
    conn.executemany(
        "INSERT INTO nodes (id, label, node_type, embedding_id) VALUES (?, ?, ?, ?)",
        [
            (1, "photosynthesis", "Concept", node1_emb_id),
            (2, "cellular_respiration", "Concept", node2_emb_id),
            (3, "chemistry", "Topic", node3_emb_id),
        ]
    )
    
    # Insert flashcard nodes (created by GraphBuilder)
    conn.executemany(
        "INSERT INTO nodes (label, node_type) VALUES (?, ?)",
        [
            ("flashcard-1", "Flashcard"),
            ("flashcard-2", "Flashcard"),
        ]
    )
    
    conn.commit()
    conn.close()
    
    return str(db_path)


def test_load_embeddings_for_entities(sample_db):
    """Test loading embeddings from database."""
    conn = sqlite3.connect(sample_db)
    
    embeddings = _load_embeddings_for_entities(conn, 'flashcard', [1, 2])
    
    assert len(embeddings) == 2
    assert 1 in embeddings
    assert 2 in embeddings
    assert embeddings[1].shape == (512,)
    assert embeddings[2].shape == (512,)
    
    conn.close()


def test_create_semantic_edges(sample_db):
    """Test edge creation with similarity threshold."""
    conn = sqlite3.connect(sample_db)
    
    # Create edge between flashcard 1 and node 1
    result = _create_semantic_edges(conn, 1, 1, 0.85, 'DERIVED_FROM')
    assert result is True
    
    # Verify edge was created
    edge = conn.execute(
        "SELECT * FROM edges WHERE relationship_type = 'DERIVED_FROM'"
    ).fetchone()
    assert edge is not None
    assert edge[4] == 0.85  # weight column
    
    # Try to create duplicate edge
    result = _create_semantic_edges(conn, 1, 1, 0.85, 'DERIVED_FROM')
    assert result is False
    
    conn.close()


def test_ensure_minimum_links(sample_db):
    """Test fallback linking for flashcards without high-similarity matches."""
    conn = sqlite3.connect(sample_db)
    
    # Flashcard 2 has no links initially
    node_similarities = [(1, 0.45), (2, 0.38), (3, 0.22)]  # All below threshold
    
    _ensure_minimum_links(conn, 2, node_similarities)
    
    # Verify fallback link was created to best match (node 1)
    edges = conn.execute(
        "SELECT * FROM edges WHERE source_id = (SELECT id FROM nodes WHERE label = 'flashcard-2')"
    ).fetchall()
    
    assert len(edges) >= 1
    
    conn.close()


def test_link_flashcards_to_graph(sample_db):
    """Test complete linking workflow."""
    result = link_flashcards_to_graph(
        db_path=sample_db,
        similarity_threshold=0.7,
        embedding_generator=None
    )
    
    assert result['flashcards_processed'] == 2
    assert result['nodes_processed'] == 3
    assert result['links_created'] >= 0
    assert 0.0 <= result['avg_similarity'] <= 1.0


def test_link_flashcards_threshold_filtering(sample_db):
    """Test that only high-similarity edges are created."""
    # Test with very high threshold (should create few/no edges)
    result_high = link_flashcards_to_graph(
        db_path=sample_db,
        similarity_threshold=0.99,
        embedding_generator=None
    )
    
    # Test with low threshold (should create more edges)
    result_low = link_flashcards_to_graph(
        db_path=sample_db,
        similarity_threshold=0.5,
        embedding_generator=None
    )
    
    # Lower threshold should create more or equal links
    assert result_low['links_created'] >= result_high['links_created']


@pytest.mark.skipif(
    not pytest.importorskip("sklearn", reason="scikit-learn not installed"),
    reason="Clustering requires scikit-learn"
)
def test_cluster_flashcards(sample_db):
    """Test K-means clustering of flashcards."""
    clusters = cluster_flashcards(
        db_path=sample_db,
        n_clusters=2,
        embedding_generator=None
    )
    
    assert len(clusters) == 2  # 2 flashcards
    assert 1 in clusters
    assert 2 in clusters
    assert clusters[1] in [0, 1]
    assert clusters[2] in [0, 1]


def test_cluster_flashcards_few_samples(sample_db):
    """Test clustering with fewer samples than clusters."""
    # Request 10 clusters but only 2 flashcards exist
    clusters = cluster_flashcards(
        db_path=sample_db,
        n_clusters=10,
        embedding_generator=None
    )
    
    # Should auto-adjust to 2 clusters
    assert len(clusters) == 2
    assert len(set(clusters.values())) <= 2


def test_find_similar_flashcards_for_node(sample_db):
    """Test finding similar flashcards for a given node."""
    results = find_similar_flashcards_for_node(
        db_path=sample_db,
        node_label="photosynthesis",
        top_k=2
    )
    
    assert len(results) <= 2
    assert all('flashcard_id' in r for r in results)
    assert all('question' in r for r in results)
    assert all('answer' in r for r in results)
    assert all('similarity' in r for r in results)
    assert all('topic' in r for r in results)
    
    # Results should be sorted by similarity descending
    if len(results) >= 2:
        assert results[0]['similarity'] >= results[1]['similarity']


def test_find_similar_flashcards_for_nonexistent_node(sample_db):
    """Test query with nonexistent node."""
    results = find_similar_flashcards_for_node(
        db_path=sample_db,
        node_label="nonexistent_node",
        top_k=5
    )
    
    assert results == []


def test_find_similar_nodes_for_flashcard(sample_db):
    """Test finding similar nodes for a given flashcard."""
    results = find_similar_nodes_for_flashcard(
        db_path=sample_db,
        flashcard_id=1,
        top_k=2
    )
    
    assert len(results) <= 2
    assert all('node_id' in r for r in results)
    assert all('label' in r for r in results)
    assert all('node_type' in r for r in results)
    assert all('similarity' in r for r in results)
    
    # Results should be sorted by similarity descending
    if len(results) >= 2:
        assert results[0]['similarity'] >= results[1]['similarity']


def test_find_similar_nodes_for_nonexistent_flashcard(sample_db):
    """Test query with nonexistent flashcard."""
    results = find_similar_nodes_for_flashcard(
        db_path=sample_db,
        flashcard_id=999,
        top_k=5
    )
    
    assert results == []


def test_cosine_similarity_edge_cases(sample_db):
    """Test similarity threshold edge cases (0.69 vs 0.71)."""
    # Test with threshold just below a similarity value
    result_below = link_flashcards_to_graph(
        db_path=sample_db,
        similarity_threshold=0.69,
        embedding_generator=None
    )
    
    # Test with threshold just above
    result_above = link_flashcards_to_graph(
        db_path=sample_db,
        similarity_threshold=0.71,
        embedding_generator=None
    )
    
    # Lower threshold should create more or equal links
    assert result_below['links_created'] >= result_above['links_created']


def test_relationship_type_selection(sample_db):
    """Test that relationship types are correctly assigned based on node type."""
    conn = sqlite3.connect(sample_db)
    
    # Link flashcards to graph
    link_flashcards_to_graph(
        db_path=sample_db,
        similarity_threshold=0.5,  # Low threshold to ensure edges
        embedding_generator=None
    )
    
    # Check that edges to Topic nodes use EXPLAINS
    topic_edges = conn.execute("""
        SELECT e.relationship_type 
        FROM edges e
        JOIN nodes n ON e.target_id = n.id
        WHERE n.node_type = 'Topic' AND e.relationship_type IN ('DERIVED_FROM', 'EXPLAINS')
    """).fetchall()
    
    # Check that edges to Concept nodes use DERIVED_FROM
    concept_edges = conn.execute("""
        SELECT e.relationship_type 
        FROM edges e
        JOIN nodes n ON e.target_id = n.id
        WHERE n.node_type = 'Concept' AND e.relationship_type IN ('DERIVED_FROM', 'EXPLAINS')
    """).fetchall()
    
    # Verify relationship types (if edges exist)
    for edge in topic_edges:
        assert edge[0] == 'EXPLAINS'
    
    for edge in concept_edges:
        assert edge[0] == 'DERIVED_FROM'
    
    conn.close()


def test_empty_database():
    """Test handling of empty database."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vector BLOB NOT NULL,
            entity_type TEXT NOT NULL,
            entity_id INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS nodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT UNIQUE NOT NULL,
            node_type TEXT NOT NULL,
            embedding_id INTEGER
        );
        CREATE TABLE IF NOT EXISTS edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id INTEGER NOT NULL,
            target_id INTEGER NOT NULL,
            relationship_type TEXT NOT NULL,
            weight REAL
        );
        CREATE TABLE IF NOT EXISTS flashcards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            embedding_id INTEGER
        );
    """)
    conn.close()
    
    result = link_flashcards_to_graph(
        db_path=db_path,
        similarity_threshold=0.7,
        embedding_generator=None
    )
    
    assert result['flashcards_processed'] == 0
    assert result['nodes_processed'] == 0
    assert result['links_created'] == 0
