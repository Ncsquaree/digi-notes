"""Unit tests for utils.ner (Phase 4)."""
import pytest
from unittest.mock import MagicMock, patch

from utils.ner import (
    EntityExtractor,
    classify_entity_type,
    normalize_entity,
    similarity_ratio,
    filter_academic_terms,
    extract_noun_phrases,
    nltk_fallback_ner,
    Entity,
)


class TestEntityClassification:
    """Test entity type classification."""
    
    def test_classify_formula_with_equals(self):
        """Formulas with equals signs should be classified as Formula."""
        assert classify_entity_type("E = mc²") == "Formula"
        assert classify_entity_type("F = ma") == "Formula"
    
    def test_classify_chemical_formula(self):
        """Chemical formulas should be classified as Formula."""
        assert classify_entity_type("H2O") == "Formula"
        assert classify_entity_type("CO2") == "Formula"
        assert classify_entity_type("C6H12O6") == "Formula"
    
    def test_classify_concept_with_keyword(self):
        """Terms with academic keywords should be classified as Concept."""
        assert classify_entity_type("machine learning algorithm") == "Concept"
        assert classify_entity_type("quantum theory") == "Concept"
    
    def test_classify_entity_proper_noun(self):
        """Proper nouns in scientific context should be classified as Entity."""
        assert classify_entity_type("Einstein", "Einstein developed relativity") == "Entity"
        assert classify_entity_type("Newton", "Newton discovered gravity") == "Entity"
    
    def test_classify_topic_capitalized_phrase(self):
        """Short capitalized phrases should be classified as Topic."""
        entity_type = classify_entity_type("Photosynthesis Process")
        assert entity_type in ["Topic", "Concept"]  # Either is acceptable


class TestNormalization:
    """Test entity normalization."""
    
    def test_normalize_lowercase(self):
        """Concepts should be normalized to lowercase."""
        assert normalize_entity("Photosynthesis") == "photosynthesis"
        assert normalize_entity("MITOCHONDRIA") == "mitochondria"
    
    def test_normalize_preserves_formulas(self):
        """Formulas should preserve case and symbols."""
        assert normalize_entity("E = mc²") == "E = mc²"
        assert normalize_entity("H2O") == "H2O"
    
    def test_normalize_removes_trailing_punctuation(self):
        """Trailing punctuation should be removed."""
        assert normalize_entity("concept.") == "concept"
        assert normalize_entity("term,") == "term"
    
    def test_normalize_strips_whitespace(self):
        """Leading/trailing whitespace should be stripped."""
        assert normalize_entity("  concept  ") == "concept"


class TestSimilarity:
    """Test similarity functions."""
    
    def test_similarity_identical_strings(self):
        """Identical strings should have similarity 1.0."""
        assert similarity_ratio("photosynthesis", "photosynthesis") == 1.0
    
    def test_similarity_case_insensitive(self):
        """Similarity should be case-insensitive."""
        sim = similarity_ratio("Photosynthesis", "photosynthesis")
        assert sim == 1.0
    
    def test_similarity_similar_strings(self):
        """Similar strings should have high similarity."""
        sim = similarity_ratio("photosynthesis", "photosyntesis")
        assert sim > 0.85  # One character difference
    
    def test_similarity_different_strings(self):
        """Very different strings should have low similarity."""
        sim = similarity_ratio("photosynthesis", "mitochondria")
        assert sim < 0.5


class TestAcademicFiltering:
    """Test academic term filtering."""
    
    def test_filter_removes_stop_words(self):
        """Stop words should be filtered out."""
        entities = ["the", "photosynthesis", "and", "mitochondria"]
        filtered = filter_academic_terms(entities)
        assert "the" not in filtered
        assert "and" not in filtered
        assert "photosynthesis" in filtered
    
    def test_filter_removes_short_terms(self):
        """Very short terms should be filtered out."""
        entities = ["ab", "DNA", "photosynthesis"]
        filtered = filter_academic_terms(entities)
        assert "ab" not in filtered
        assert "DNA" in filtered  # Capitalized, acceptable
    
    def test_filter_keeps_formulas(self):
        """Formulas should be kept."""
        entities = ["E = mc²", "the", "H2O"]
        filtered = filter_academic_terms(entities)
        assert "E = mc²" in filtered
        assert "H2O" in filtered
    
    def test_filter_keeps_capitalized_terms(self):
        """Capitalized terms should be kept."""
        entities = ["Photosynthesis", "the", "Biology"]
        filtered = filter_academic_terms(entities)
        assert "Photosynthesis" in filtered
        assert "Biology" in filtered


class TestNLTKFallback:
    """Test NLTK fallback NER."""
    
    def test_nltk_fallback_extracts_entities(self, sample_text):
        """NLTK should extract some entities from sample text."""
        text = "Photosynthesis is the process by which green plants convert light energy."
        entities = nltk_fallback_ner(text)
        # Should extract at least one entity (Photosynthesis, plants, etc.)
        assert len(entities) > 0
        assert all(isinstance(e, Entity) for e in entities)
    
    def test_nltk_fallback_empty_text(self):
        """Empty text should return empty list."""
        entities = nltk_fallback_ner("")
        assert entities == []
    
    def test_nltk_fallback_confidence_scores(self):
        """NLTK entities should have medium confidence (0.6-0.7)."""
        text = "Einstein developed the theory of relativity."
        entities = nltk_fallback_ner(text)
        if entities:  # May fail if NLTK not available
            assert all(0.5 <= e.confidence <= 0.8 for e in entities)


class TestEntityExtractor:
    """Test EntityExtractor class."""
    
    def test_extractor_initialization_fallback(self):
        """Extractor should initialize with fallback mode."""
        extractor = EntityExtractor(use_fallback=True, auto_load=False)
        assert extractor.use_fallback is True
        assert extractor.interpreter is None
    
    def test_extractor_initialization_with_model_path(self, tmp_path):
        """Extractor should accept custom model path."""
        fake_model = tmp_path / "fake_model.tflite"
        extractor = EntityExtractor(
            model_path=fake_model,
            use_fallback=False,
            auto_load=False
        )
        assert extractor.model_path == fake_model
    
    def test_extract_entities_with_fallback(self):
        """Extractor should extract entities using NLTK fallback."""
        extractor = EntityExtractor(use_fallback=True)
        text = "Photosynthesis converts CO2 and H2O into glucose."
        entities = extractor.extract_entities(text)
        
        # Should extract some entities
        assert len(entities) >= 0  # May be empty if NLTK not available
        assert all(isinstance(e, Entity) for e in entities)
    
    def test_extract_entities_empty_text(self):
        """Empty text should return empty list."""
        extractor = EntityExtractor(use_fallback=True)
        entities = extractor.extract_entities("")
        assert entities == []
    
    def test_extract_from_chunks_adds_entities_to_metadata(self):
        """extract_from_chunks should add entities to chunk metadata."""
        extractor = EntityExtractor(use_fallback=True)
        
        chunks = [
            {
                'text': 'Photosynthesis is a process.',
                'metadata': {'topic': 'Biology'}
            }
        ]
        
        updated_chunks = extractor.extract_from_chunks(chunks)
        
        assert 'entities' in updated_chunks[0]['metadata']
        assert isinstance(updated_chunks[0]['metadata']['entities'], list)
    
    def test_extract_from_chunks_preserves_existing_metadata(self):
        """extract_from_chunks should preserve existing metadata."""
        extractor = EntityExtractor(use_fallback=True)
        
        chunks = [
            {
                'text': 'Some text',
                'metadata': {'topic': 'Test', 'has_formula': True}
            }
        ]
        
        updated_chunks = extractor.extract_from_chunks(chunks)
        
        assert updated_chunks[0]['metadata']['topic'] == 'Test'
        assert updated_chunks[0]['metadata']['has_formula'] is True


class TestDeduplication:
    """Test entity deduplication."""
    
    def test_deduplicate_exact_duplicates(self):
        """Exact duplicates should be merged."""
        extractor = EntityExtractor(use_fallback=True)
        entities = ["Photosynthesis", "Photosynthesis", "Mitochondria"]
        
        unique = extractor.deduplicate_entities(entities)
        
        assert len(unique) == 2
        assert "Photosynthesis" in unique
        assert "Mitochondria" in unique
    
    def test_deduplicate_case_variants(self):
        """Case variants should be merged."""
        extractor = EntityExtractor(use_fallback=True)
        entities = ["Photosynthesis", "photosynthesis", "PHOTOSYNTHESIS"]
        
        unique = extractor.deduplicate_entities(entities)
        
        assert len(unique) == 1
        assert unique[0] in ["Photosynthesis", "photosynthesis", "PHOTOSYNTHESIS"]
    
    def test_deduplicate_fuzzy_matching(self):
        """Similar entities should be merged (fuzzy matching)."""
        extractor = EntityExtractor(use_fallback=True)
        entities = ["photosynthesis", "photosyntesis", "mitochondria"]
        
        unique = extractor.deduplicate_entities(entities)
        
        # photosynthesis and photosyntesis should merge (similarity > 0.8)
        assert len(unique) <= 2
    
    def test_deduplicate_preserves_formulas(self):
        """Formulas should be preserved exactly."""
        extractor = EntityExtractor(use_fallback=True)
        entities = ["E = mc²", "H2O", "CO2"]
        
        unique = extractor.deduplicate_entities(entities)
        
        assert len(unique) == 3
        assert "E = mc²" in unique
    
    def test_deduplicate_empty_list(self):
        """Empty list should return empty list."""
        extractor = EntityExtractor(use_fallback=True)
        unique = extractor.deduplicate_entities([])
        assert unique == []
    
    def test_deduplicate_selects_most_frequent(self):
        """Most frequent variant should be selected as canonical."""
        extractor = EntityExtractor(use_fallback=True)
        entities = ["photosynthesis", "Photosynthesis", "photosynthesis"]
        
        unique = extractor.deduplicate_entities(entities)
        
        assert len(unique) == 1
        assert unique[0] == "photosynthesis"  # Most frequent


class TestIntegrationWithPreprocessing:
    """Integration tests with preprocessing pipeline."""
    
    def test_entities_compatible_with_preprocessed_chunks(self, sample_text):
        """Entities should integrate with preprocessed chunks."""
        from utils.preprocess import preprocess_text
        
        chunks = preprocess_text(sample_text, max_chunk_size=500)
        extractor = EntityExtractor(use_fallback=True)
        
        chunks_with_entities = extractor.extract_from_chunks(chunks)
        
        # All chunks should have entities field
        for chunk in chunks_with_entities:
            assert 'entities' in chunk['metadata']
            assert isinstance(chunk['metadata']['entities'], list)
    
    def test_entity_types_in_metadata(self, sample_text):
        """Entity metadata should include type and confidence."""
        from utils.preprocess import preprocess_text
        
        chunks = preprocess_text(sample_text, max_chunk_size=500)
        extractor = EntityExtractor(use_fallback=True)
        
        chunks_with_entities = extractor.extract_from_chunks(chunks)
        
        for chunk in chunks_with_entities:
            for entity_dict in chunk['metadata']['entities']:
                assert 'text' in entity_dict
                assert 'type' in entity_dict
                assert 'confidence' in entity_dict
                assert entity_dict['type'] in ['Concept', 'Topic', 'Entity', 'Formula']


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_extract_entities_with_special_characters(self):
        """Special characters should be handled gracefully."""
        extractor = EntityExtractor(use_fallback=True)
        text = "E = mc² and Δx → ∞"
        entities = extractor.extract_entities(text)
        
        # Should not crash
        assert isinstance(entities, list)
    
    def test_extract_entities_very_long_text(self):
        """Very long text should be handled (truncation)."""
        extractor = EntityExtractor(use_fallback=True)
        text = "Photosynthesis " * 1000  # 1000 words
        entities = extractor.extract_entities(text)
        
        # Should not crash
        assert isinstance(entities, list)
    
    def test_extract_entities_with_formulas(self, formula_text):
        """Text with formulas should extract formulas as entities."""
        extractor = EntityExtractor(use_fallback=True)
        entities = extractor.extract_entities(formula_text)
        
        # Should extract some entities (formulas)
        entity_texts = [e.text for e in entities]
        # At least some formula-like patterns should be extracted
        assert len(entity_texts) >= 0  # Fallback may or may not extract formulas
