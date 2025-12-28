"""Unit tests for utils.embeddings (Phase 3)."""
import numpy as np
import pytest
import sqlite3

from utils.embeddings import (
    EmbeddingGenerator,
    cosine_similarity,
    generate_embedding,
)


def test_generate_embedding_fallback_returns_unit_vector(sample_text):
    text = sample_text.split("\n\n")[0]
    vec = generate_embedding(text)
    assert len(vec) == 512
    assert pytest.approx(1.0, rel=1e-3) == float(np.linalg.norm(vec))


def test_generate_embedding_returns_zero_for_empty():
    vec = generate_embedding("   ")
    assert vec == [0.0] * 512


def test_cosine_similarity_handles_zero_vector():
    a = [1.0, 0.0, 0.0]
    b = [0.0, 0.0, 0.0]
    assert cosine_similarity(a, b) == 0.0


def test_cosine_similarity_self_is_one():
    a = [0.2, 0.2, 0.2]
    assert pytest.approx(1.0) == cosine_similarity(a, a)


def test_store_and_fetch_embedding(tmp_path):
    db_path = tmp_path / "embeddings.db"
    # Initialize schema manually for test (mimics init_db.py)
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vector BLOB NOT NULL,
            entity_type TEXT NOT NULL,
            entity_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()
    
    gen = EmbeddingGenerator(db_path=str(db_path), force_hash_fallback=True)

    vec = gen.embed_text("hello world")
    embedding_id = gen.store_embedding("chunk-1", "hello world", vec)

    loaded = gen.fetch_embedding(embedding_id)
    assert loaded is not None
    assert len(loaded) == len(vec)
    assert np.allclose(loaded, vec)


def test_embed_and_store_chunks_creates_ids(tmp_path):
    db_path = tmp_path / "embeddings.db"
    # Initialize schema
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vector BLOB NOT NULL,
            entity_type TEXT NOT NULL,
            entity_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()
    
    gen = EmbeddingGenerator(db_path=str(db_path), force_hash_fallback=True)

    chunks = [
        {"id": "c1", "text": "alpha"},
        {"text": "beta"},
    ]

    results = gen.embed_and_store_chunks(chunks)
    assert len(results) == 2
    assert results[0].chunk_id == "c1"
    assert results[1].chunk_id == "chunk-1"

    # Check embedding_id in metadata
    assert "embedding_id" in results[0].metadata
    stored = gen.fetch_embedding(results[0].metadata["embedding_id"])
    assert stored is not None


def test_similarity_search_orders_results(tmp_path):
    db_path = tmp_path / "embeddings.db"
    # Initialize schema
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vector BLOB NOT NULL,
            entity_type TEXT NOT NULL,
            entity_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()
    
    gen = EmbeddingGenerator(db_path=str(db_path), force_hash_fallback=True)

    # Store two distinct embeddings
    gen.store_embedding("earth", "Earth is the third planet.", gen.embed_text("Earth is the third planet."))
    gen.store_embedding("mars", "Mars is a red planet.", gen.embed_text("Mars is a red planet."))

    results = gen.similarity_search("Earth planet", top_k=2)
    assert len(results) == 2
    assert "entity_type" in results[0]
    assert results[0]["entity_type"] == "chunk"
    # First result should have highest score
    assert results[0]["score"] >= results[1]["score"]
