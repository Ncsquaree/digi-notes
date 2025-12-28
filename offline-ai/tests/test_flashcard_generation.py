"""Unit tests for flashcards.generate (Phase 5)."""
import pytest
import sqlite3
from unittest.mock import MagicMock

from flashcards.generate import (
    FlashcardGenerator,
    Flashcard,
    QUESTION_TEMPLATES,
)


class TestQuestionGeneration:
    """Test question template selection and generation."""
    
    def test_select_templates_formula_chunks(self):
        """Formula chunks should use formula templates."""
        gen = FlashcardGenerator(db_path=":memory:")
        chunk = {
            'metadata': {
                'has_formula': True,
                'entities': [],
                'type': 'formula'
            }
        }
        
        templates = gen._select_templates(chunk)
        assert len(templates) > 0
        # Should contain formula templates
        assert any('formula' in t.lower() for t in templates)
    
    def test_select_templates_multi_entity(self):
        """Multi-entity chunks should use difference templates."""
        gen = FlashcardGenerator(db_path=":memory:")
        chunk = {
            'metadata': {
                'has_formula': False,
                'entities': [{'text': 'Entity1'}, {'text': 'Entity2'}],
                'type': 'paragraph'
            }
        }
        
        templates = gen._select_templates(chunk)
        assert len(templates) > 0
    
    def test_generate_rule_based_questions(self):
        """Should generate questions from templates."""
        gen = FlashcardGenerator(db_path=":memory:")
        chunk = {
            'text': 'Photosynthesis converts light to chemical energy.',
            'metadata': {
                'topic': 'Photosynthesis',
                'entities': [{'text': 'Photosynthesis', 'type': 'Concept'}],
                'type': 'paragraph',
                'has_formula': False,
                'academic_level': 'basic'
            }
        }
        
        questions = gen._generate_rule_based_questions(chunk)
        assert len(questions) > 0
        assert all(isinstance(q, str) for q in questions)
        # Questions should not have dangling placeholders
        assert all('{' not in q and '}' not in q for q in questions)


class TestAnswerExtraction:
    """Test answer extraction and validation."""
    
    def test_validate_answer_valid(self):
        """Valid answers should pass validation."""
        gen = FlashcardGenerator(db_path=":memory:")
        
        valid_answers = [
            "The process of converting light energy to chemical energy",
            "A complex biological mechanism involving multiple steps",
            "The rate of change of velocity with respect to time"
        ]
        
        for answer in valid_answers:
            assert gen._validate_answer(answer) is True
    
    def test_validate_answer_too_short(self):
        """Answers < 10 chars should fail validation."""
        gen = FlashcardGenerator(db_path=":memory:")
        assert gen._validate_answer("Too short") is False
    
    def test_validate_answer_too_long(self):
        """Answers > 200 chars should fail validation."""
        gen = FlashcardGenerator(db_path=":memory:")
        long_answer = "A" * 201
        assert gen._validate_answer(long_answer) is False
    
    def test_extract_heuristic_what_is(self):
        """Heuristic extraction should handle 'What is' questions."""
        gen = FlashcardGenerator(db_path=":memory:")
        
        question = "What is photosynthesis?"
        text = "Photosynthesis is a biological process that converts light energy into chemical energy"
        
        result = gen._extract_heuristic(question, text)
        assert result is not None
        answer, confidence = result
        assert len(answer) > 0
        assert 0 <= confidence <= 1.0


class TestDifficultMapping:
    """Test academic level to difficulty mapping."""
    
    def test_map_difficulty_basic(self):
        """Basic level should map to difficulty 1-2."""
        gen = FlashcardGenerator(db_path=":memory:")
        metadata = {'academic_level': 'basic', 'entities': [], 'has_formula': False}
        difficulty = gen._map_difficulty(metadata)
        assert difficulty in [1, 2]
    
    def test_map_difficulty_intermediate(self):
        """Intermediate level should map to difficulty 2-3."""
        gen = FlashcardGenerator(db_path=":memory:")
        metadata = {'academic_level': 'intermediate', 'entities': [], 'has_formula': False}
        difficulty = gen._map_difficulty(metadata)
        assert 2 <= difficulty <= 3
    
    def test_map_difficulty_advanced(self):
        """Advanced level should map to difficulty 4-5."""
        gen = FlashcardGenerator(db_path=":memory:")
        metadata = {'academic_level': 'advanced', 'entities': [], 'has_formula': False}
        difficulty = gen._map_difficulty(metadata)
        assert difficulty >= 4
    
    def test_map_difficulty_with_formula(self):
        """Formulas should increase difficulty."""
        gen = FlashcardGenerator(db_path=":memory:")
        metadata_no_formula = {'academic_level': 'basic', 'entities': [], 'has_formula': False}
        metadata_with_formula = {'academic_level': 'basic', 'entities': [], 'has_formula': True}
        
        d1 = gen._map_difficulty(metadata_no_formula)
        d2 = gen._map_difficulty(metadata_with_formula)
        
        assert d2 > d1


class TestDeduplication:
    """Test deduplication and similarity."""
    
    def test_normalize_question(self):
        """Questions should normalize to lowercase, no punctuation."""
        gen = FlashcardGenerator(db_path=":memory:")
        
        q1 = "What is Photosynthesis?"
        q2 = "what is photosynthesis"
        
        n1 = gen._normalize_question(q1)
        n2 = gen._normalize_question(q2)
        
        assert n1 == n2
    
    def test_similarity_identical_questions(self):
        """Identical questions should have high similarity."""
        gen = FlashcardGenerator(db_path=":memory:")
        
        q1 = "What is photosynthesis?"
        q2 = "What is photosynthesis?"
        
        similarity = gen._similarity_ratio(q1, q2)
        assert similarity == 1.0
    
    def test_similarity_different_questions(self):
        """Very different questions should have low similarity."""
        gen = FlashcardGenerator(db_path=":memory:")
        
        q1 = "What is photosynthesis?"
        q2 = "How do mitochondria function?"
        
        similarity = gen._similarity_ratio(q1, q2)
        assert similarity < 0.5
    
    def test_deduplicate_flashcards(self):
        """Identical or similar flashcards should be deduplicated."""
        gen = FlashcardGenerator(db_path=":memory:")
        
        cards = [
            Flashcard(
                question="What is photosynthesis?",
                answer="Process that converts light to chemical energy",
                topic="Biology",
                entities=[],
                embedding_id=1,
                difficulty=2,
                source_type="rule_based"
            ),
            Flashcard(
                question="What is photosynthesis?",  # Duplicate
                answer="Same as above",
                topic="Biology",
                entities=[],
                embedding_id=1,
                difficulty=2,
                source_type="rule_based"
            ),
            Flashcard(
                question="How do plants use photosynthesis?",  # Similar
                answer="Plants use it to produce energy",
                topic="Biology",
                entities=[],
                embedding_id=2,
                difficulty=3,
                source_type="rule_based"
            )
        ]
        
        unique = gen._deduplicate_flashcards(cards)
        
        # Should have fewer than original
        assert len(unique) < len(cards)


class TestFlashcardGeneration:
    """Test flashcard generation pipeline."""
    
    def test_generate_from_chunk(self):
        """Should generate flashcards from a single chunk."""
        gen = FlashcardGenerator(db_path=":memory:", ner_extractor=None)
        
        chunk = {
            'text': 'Photosynthesis is the process by which plants convert light energy into chemical energy using chlorophyll.',
            'metadata': {
                'topic': 'Photosynthesis',
                'entities': [{'text': 'Photosynthesis', 'type': 'Concept'}],
                'type': 'paragraph',
                'has_formula': False,
                'academic_level': 'basic',
                'embedding_id': 1
            }
        }
        
        flashcards = gen.generate_from_chunk(chunk)
        
        assert len(flashcards) > 0
        assert all(isinstance(fc, Flashcard) for fc in flashcards)
        assert all(fc.topic == 'Photosynthesis' for fc in flashcards)
        assert all(fc.embedding_id == 1 for fc in flashcards)
    
    def test_generate_from_chunk_empty(self):
        """Empty chunk should return empty flashcards."""
        gen = FlashcardGenerator(db_path=":memory:")
        chunk = {'text': '', 'metadata': {}}
        
        flashcards = gen.generate_from_chunk(chunk)
        assert flashcards == []
    
    def test_generate_from_chunks(self, sample_text):
        """Should generate flashcards from multiple chunks."""
        from utils.preprocess import preprocess_text
        
        gen = FlashcardGenerator(db_path=":memory:", ner_extractor=None)
        
        chunks = preprocess_text(sample_text, max_chunk_size=500)
        flashcards = gen.generate_from_chunks(chunks)
        
        # Should generate some flashcards
        assert len(flashcards) > 0
        assert all(isinstance(fc, Flashcard) for fc in flashcards)


class TestDatabaseStorage:
    """Test SQLite storage."""
    
    def test_store_flashcards(self, tmp_path):
        """Should store flashcards in SQLite."""
        db_path = tmp_path / "test.db"
        
        # Initialize schema
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS flashcards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                difficulty INTEGER DEFAULT 0,
                context TEXT,
                source_type TEXT DEFAULT 'generated',
                embedding_id INTEGER
            );
        """)
        conn.commit()
        conn.close()
        
        gen = FlashcardGenerator(db_path=str(db_path))
        
        flashcards = [
            Flashcard(
                question="What is photosynthesis?",
                answer="Process that converts light to chemical energy",
                topic="Biology",
                entities=["photosynthesis"],
                embedding_id=1,
                difficulty=2,
                source_type="rule_based"
            ),
            Flashcard(
                question="How does photosynthesis work?",
                answer="Plants use chlorophyll to capture light energy",
                topic="Biology",
                entities=["photosynthesis", "chlorophyll"],
                embedding_id=2,
                difficulty=3,
                source_type="qa_extracted"
            )
        ]
        
        inserted_ids = gen.store_flashcards(flashcards)
        
        assert len(inserted_ids) == 2
        assert all(isinstance(id, int) and id > 0 for id in inserted_ids)
        
        # Verify stored in database
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("SELECT COUNT(*) FROM flashcards")
        count = cursor.fetchone()[0]
        conn.close()
        
        assert count == 2


class TestIntegration:
    """Integration tests with preprocessing pipeline."""
    
    def test_full_flashcard_pipeline(self, sample_text):
        """Test full pipeline: preprocess → embeddings → NER → flashcards."""
        from utils.preprocess import preprocess_text
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            # Initialize schema
            conn = sqlite3.connect(db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS flashcards (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    difficulty INTEGER DEFAULT 0,
                    context TEXT,
                    source_type TEXT DEFAULT 'generated',
                    embedding_id INTEGER
                );
            """)
            conn.commit()
            conn.close()
            
            # Preprocess
            chunks = preprocess_text(sample_text, max_chunk_size=500)
            
            # Simulate NER (add entities to metadata)
            for chunk in chunks:
                chunk['metadata']['entities'] = [
                    {'text': 'Photosynthesis', 'type': 'Concept'},
                    {'text': 'ATP', 'type': 'Concept'}
                ]
                chunk['metadata']['embedding_id'] = 1
            
            # Generate flashcards
            gen = FlashcardGenerator(db_path=db_path, ner_extractor=None)
            flashcards = gen.generate_from_chunks(chunks)
            gen.store_flashcards(flashcards)
            
            # Verify generation
            assert len(flashcards) > 0
            for fc in flashcards:
                assert fc.question
                assert fc.answer
                assert fc.topic
        
        finally:
            import os
            if os.path.exists(db_path):
                os.remove(db_path)
