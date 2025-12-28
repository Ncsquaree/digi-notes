#!/usr/bin/env python3
"""Verify offline-ai setup is complete."""
import os
import sqlite3
import sys

def check_models():
    """Verify TFLite models exist."""
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    required = ['mobilebert-squad.tflite', 'use-lite.tflite']
    missing = [m for m in required if not os.path.exists(os.path.join(models_dir, m))]
    if missing:
        print(f"✗ Missing models: {', '.join(missing)}")
        print("  Run: python scripts/download_models.py")
        return False
    print(f"✓ Models found: {', '.join(required)}")
    return True

def check_database():
    """Verify SQLite database exists and has tables."""
    db_path = os.path.join(os.path.dirname(__file__), '..', 'offline_ai.db')
    if not os.path.exists(db_path):
        print("✗ Database not found")
        print("  Run: python scripts/init_db.py")
        return False
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    required = ['nodes', 'edges', 'embeddings', 'flashcards']
    missing = [t for t in required if t not in tables]
    conn.close()
    
    if missing:
        print(f"✗ Missing tables: {', '.join(missing)}")
        return False
    print(f"✓ Database tables: {', '.join(required)}")
    return True

def check_dependencies():
    """Verify Python packages are installed."""
    try:
        import tensorflow
        import numpy
        import nltk
        import sentencepiece
        print("✓ Dependencies installed")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("  Run: pip install -r requirements.txt")
        return False

def main():
    print("=== Offline AI Setup Verification ===\n")
    checks = [
        check_dependencies(),
        check_models(),
        check_database()
    ]
    
    if all(checks):
        print("\n✓ Setup complete! Ready for Phase 2 implementation.")
        sys.exit(0)
    else:
        print("\n✗ Setup incomplete. Fix errors above.")
        sys.exit(1)

if __name__ == '__main__':
    main()
