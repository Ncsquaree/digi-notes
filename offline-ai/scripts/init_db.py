#!/usr/bin/env python3
"""
Initialize SQLite database for offline AI system.
Schema:
  - nodes: Knowledge graph vertices (Topic/Concept/Entity/Flashcard)
  - edges: Relationships between nodes
  - embeddings: Vector representations for semantic search
  - flashcards: Generated Q&A pairs with metadata

Usage:
  python init_db.py          # Create database (preserve existing data)
  python init_db.py --reset  # Delete and recreate database
"""
import sqlite3
import os
import json
import sys
import argparse

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'offline_ai.db')

SCHEMA = """
-- Knowledge Graph Nodes
CREATE TABLE IF NOT EXISTS nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_type TEXT NOT NULL,           -- Topic, Concept, Entity, Flashcard
    label TEXT NOT NULL,                -- Human-readable name
    properties TEXT,                    -- JSON: {description, examples, etc.}
    embedding_id INTEGER,               -- FK to embeddings table
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (embedding_id) REFERENCES embeddings(id)
);

-- Knowledge Graph Edges
CREATE TABLE IF NOT EXISTS edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL,
    target_id INTEGER NOT NULL,
    relationship_type TEXT NOT NULL,    -- RELATED_TO, PART_OF, EXPLAINS, DERIVED_FROM
    weight REAL DEFAULT 1.0,            -- Relationship strength (0-1)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_id) REFERENCES nodes(id),
    FOREIGN KEY (target_id) REFERENCES nodes(id)
);

-- Embeddings (512-dim vectors from USE Lite)
CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vector BLOB NOT NULL,               -- Serialized numpy array (512 floats)
    entity_type TEXT NOT NULL,          -- chunk, flashcard, concept, etc.
    entity_id INTEGER,                  -- FK to nodes or flashcards
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Flashcards
CREATE TABLE IF NOT EXISTS flashcards (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    difficulty INTEGER DEFAULT 0,       -- 0-5 scale
    context TEXT,                       -- Topic/source context
    source_type TEXT DEFAULT 'generated', -- parsed_question, generated, concept, formula
    embedding_id INTEGER,               -- FK to embeddings
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (embedding_id) REFERENCES embeddings(id)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_nodes_label ON nodes(label);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(relationship_type);
CREATE INDEX IF NOT EXISTS idx_embeddings_entity ON embeddings(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_flashcards_difficulty ON flashcards(difficulty);
"""

def init_database(reset: bool = False):
    """Create database and tables."""
    print(f"Initializing database: {DB_PATH}")
    
    # Optionally reset database (delete only if --reset flag provided)
    if reset:
        if os.path.exists(DB_PATH):
            print(f"⚠ Resetting database (--reset flag detected)...")
            os.remove(DB_PATH)
        else:
            print(f"ℹ Database does not exist; creating new one...")
    else:
        if os.path.exists(DB_PATH):
            print(f"ℹ Database already exists; preserving existing data...")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Execute schema (CREATE TABLE IF NOT EXISTS will skip if tables exist)
    cursor.executescript(SCHEMA)
    conn.commit()
    
    # Insert sample data only if tables are empty (new database)
    cursor.execute("SELECT COUNT(*) FROM nodes")
    node_count = cursor.fetchone()[0]
    
    if node_count == 0 and reset:
        print("ℹ Inserting sample data for new database...")
        sample_node = {
            'node_type': 'Concept',
            'label': 'Photosynthesis',
            'properties': json.dumps({
                'definition': 'Process by which plants convert light energy to chemical energy',
                'examples': ['Chloroplast reactions', 'Light-dependent reactions']
            })
        }
        cursor.execute(
            "INSERT INTO nodes (node_type, label, properties) VALUES (?, ?, ?)",
            (sample_node['node_type'], sample_node['label'], sample_node['properties'])
        )
        
        sample_flashcard = {
            'question': 'What is photosynthesis?',
            'answer': 'The process by which plants convert light energy into chemical energy',
            'difficulty': 1,
            'context': 'Biology - Plant Processes'
        }
        cursor.execute(
            "INSERT INTO flashcards (question, answer, difficulty, context) VALUES (?, ?, ?, ?)",
            (sample_flashcard['question'], sample_flashcard['answer'], 
             sample_flashcard['difficulty'], sample_flashcard['context'])
        )
        
        conn.commit()
    
    # Verify tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"✓ Tables available: {', '.join(tables)}")
    
    # Show data counts
    cursor.execute("SELECT COUNT(*) FROM nodes")
    node_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM flashcards")
    flashcard_count = cursor.fetchone()[0]
    print(f"✓ Database records: {node_count} nodes, {flashcard_count} flashcards")
    
    conn.close()
    print(f"✓ Database ready: {DB_PATH}")

def main():
    parser = argparse.ArgumentParser(description="Initialize offline AI database")
    parser.add_argument('--reset', action='store_true', help='Delete and recreate database')
    args = parser.parse_args()
    
    init_database(reset=args.reset)

if __name__ == '__main__':
    main()
