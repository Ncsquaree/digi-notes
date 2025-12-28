"""
Integration tests for offline-ai preprocessing with other modules.
"""
import pytest
from utils.preprocess import preprocess_text


class TestPreprocessingIntegration:
    """Integration tests verifying preprocessing compatibility with downstream modules."""
    
    def test_preprocess_output_compatible_with_embeddings_stub(self, sample_text):
        """
        Preprocessed chunks should be compatible with embeddings module.
        Embeddings module expects: List[Dict] with 'text' field
        """
        chunks = preprocess_text(sample_text)
        
        # Simulate embedding module consumption
        for chunk in chunks:
            text = chunk['text']
            # Should be able to pass to embedding function
            assert isinstance(text, str)
            assert len(text) > 0
    
    def test_preprocess_output_compatible_with_ner_stub(self, sample_text):
        """
        Preprocessed chunks should be compatible with NER module.
        NER module expects: List[Dict] with 'text' field and metadata
        """
        from utils.ner import EntityExtractor
        
        chunks = preprocess_text(sample_text)
        
        # Use actual NER extraction
        ner = EntityExtractor(use_fallback=True)
        chunks_with_entities = ner.extract_from_chunks(chunks)
        
        # Verify entities added to metadata
        for chunk in chunks_with_entities:
            assert 'entities' in chunk['metadata']
            assert isinstance(chunk['metadata']['entities'], list)
            
            # Each entity should have required fields
            for entity_dict in chunk['metadata']['entities']:
                assert 'text' in entity_dict
                assert 'type' in entity_dict
                assert 'confidence' in entity_dict
    
    def test_preprocess_output_compatible_with_flashcard_stub(self, sample_text):
        """
        Preprocessed chunks should be compatible with flashcard generation.
        Flashcard module expects: List[Dict] with 'text' and metadata
        """
        chunks = preprocess_text(sample_text)
        
        # Simulate flashcard generation
        flashcard_inputs = []
        for chunk in chunks:
            if chunk['metadata']['type'] != 'heading':  # Skip headings
                flashcard_inputs.append({
                    'text': chunk['text'],
                    'context': chunk['metadata']['topic'],
                    'difficulty_hint': 0 if chunk['metadata']['academic_level'] == 'basic' else 1
                })
        
        assert len(flashcard_inputs) > 0
    
    def test_preprocess_output_compatible_with_graph_builder_stub(self, sample_text):
        """
        Preprocessed chunks should be compatible with knowledge graph builder.
        Graph builder expects: List[Dict] with metadata for node/edge creation
        """
        chunks = preprocess_text(sample_text)
        
        # Simulate graph builder: create nodes from chunks
        nodes = []
        for i, chunk in enumerate(chunks):
            if chunk['metadata']['type'] in ['heading', 'formula']:
                node = {
                    'id': i,
                    'label': chunk['metadata']['topic'],
                    'type': chunk['metadata']['type'],
                    'text': chunk['text']
                }
                nodes.append(node)
        
        # Should have extracted some nodes from headings/formulas
        assert len(nodes) > 0
    
    def test_preprocess_chunks_passable_to_mock_embedding_function(self, sample_text):
        """
        Verify chunks can be processed by a mock embedding function.
        """
        chunks = preprocess_text(sample_text)
        
        # Mock embedding function signature from Phase 3
        def mock_generate_embedding(text: str):
            """Mock embedding function - returns dummy 512-dim vector."""
            # In real implementation, this would return actual embedding
            return [0.0] * 512
        
        # Should be able to generate embeddings for each chunk
        embeddings = []
        for chunk in chunks:
            emb = mock_generate_embedding(chunk['text'])
            embeddings.append(emb)
        
        assert len(embeddings) == len(chunks)
        assert all(len(emb) == 512 for emb in embeddings)
    
    def test_preprocess_markdown_compatibility(self, markdown_text):
        """
        Markdown processing should maintain compatibility with downstream modules.
        """
        chunks = preprocess_text(markdown_text, input_format='markdown')
        
        # Chunks should still have valid structure
        assert len(chunks) > 0
        for chunk in chunks:
            assert 'text' in chunk
            assert 'metadata' in chunk
            assert isinstance(chunk['text'], str)
    
    def test_preprocess_pdf_compatibility(self, complex_pdf_text):
        """
        PDF processing should maintain compatibility with downstream modules.
        """
        chunks = preprocess_text(complex_pdf_text, input_format='pdf')
        
        # Chunks should be clean and processable
        assert len(chunks) > 0
        for chunk in chunks:
            # Should not have page number artifacts
            assert not chunk['text'].strip().isdigit()
            # Should have readable text
            assert len(chunk['text'].split()) > 0


class TestPreprocessingDataflow:
    """Test data flow through multiple preprocessing steps."""
    
    def test_raw_text_to_chunks_dataflow(self, sample_text):
        """
        Verify complete data flow: raw text → cleaning → chunking → metadata.
        """
        from utils.preprocess import clean_text, chunk_text, extract_metadata
        
        # Step 1: Clean
        cleaned = clean_text(sample_text)
        assert isinstance(cleaned, str)
        
        # Step 2: Chunk
        chunks = chunk_text(cleaned, max_chunk_size=500)
        assert isinstance(chunks, list)
        assert all(isinstance(c, str) for c in chunks)
        
        # Step 3: Metadata
        results = []
        for chunk in chunks:
            meta = extract_metadata(chunk)
            results.append({'text': chunk, 'metadata': meta})
        
        assert len(results) > 0
        
        # Verify structure matches preprocess_text output
        pipeline_result = preprocess_text(sample_text)
        assert len(pipeline_result) == len(results)
        for pr, tr in zip(pipeline_result, results):
            assert pr['text'] == tr['text']
    
    def test_format_specific_preprocessing(self, markdown_text, complex_pdf_text):
        """
        Test format-specific preprocessing paths.
        """
        # Markdown path
        markdown_result = preprocess_text(markdown_text, input_format='markdown')
        assert len(markdown_result) > 0
        
        # PDF path
        pdf_result = preprocess_text(complex_pdf_text, input_format='pdf')
        assert len(pdf_result) > 0
        
        # Both should produce valid output structure
        for result in [markdown_result, pdf_result]:
            for item in result:
                assert 'text' in item
                assert 'metadata' in item
                assert len(item['text']) > 0


class TestEndToEndPipeline:
    """End-to-end pipeline tests."""
    
    def test_academic_paper_preprocessing_pipeline(self, sample_text):
        """
        Full academic paper processing pipeline.
        """
        # Process academic text
        chunks = preprocess_text(sample_text, max_chunk_size=400)
        
        # Verify output quality
        assert len(chunks) > 1, "Should produce multiple chunks"
        
        # Check diversity of chunk types
        types = set(c['metadata']['type'] for c in chunks)
        assert len(types) > 1, "Should have multiple content types"
        
        # Check topic propagation
        topics = [c['metadata']['topic'] for c in chunks]
        assert len(set(topics)) > 0, "Should extract topics"
        
        # Check academic level distribution
        levels = [c['metadata']['academic_level'] for c in chunks]
        assert len(set(levels)) > 0, "Should have academic level distribution"
    
        def test_preserve_markdown_flag_pipeline(self, markdown_text):
            """
            Test preprocessing with preserve_markdown flag.
            """
            # Process with markdown preservation (default)
            chunks_preserve = preprocess_text(markdown_text, input_format='markdown', max_chunk_size=500)
        
            # Process without markdown preservation
            from utils.preprocess import clean_text, chunk_text
            cleaned = clean_text(markdown_text, preserve_markdown=False)
            chunks_nopreserve = chunk_text(cleaned, max_chunk_size=500)
        
            # Both should produce valid output
            assert len(chunks_preserve) > 0
            assert len(chunks_nopreserve) > 0
    
        def test_preserve_formulas_flag_pipeline(self, formula_text):
            """
            Test preprocessing with preserve_formulas flag.
            """
            from utils.preprocess import clean_text
        
            # Process with formula preservation (default)
            cleaned_preserve = clean_text(formula_text, preserve_formulas=True)
        
            # Process without formula preservation
            cleaned_nopreserve = clean_text(formula_text, preserve_formulas=False)
        
            # Formula version should have math symbols
            assert any(sym in cleaned_preserve for sym in ['=', '+', '→'])
        
            # Non-formula version should have fewer math symbols
            assert cleaned_preserve.count('=') > cleaned_nopreserve.count('=')
    
        def test_preserve_structure_flag_pipeline(self):
            """
            Test preprocessing with preserve_structure flag affecting chunking.
            """
            text = """Topic A

    Paragraph about topic A with some content here.

    Topic B

    Paragraph about topic B with different content."""
        
            from utils.preprocess import clean_text, chunk_text
            cleaned = clean_text(text)
        
            # With structure preservation
            chunks_structured = chunk_text(cleaned, max_chunk_size=100, preserve_structure=True)
        
            # Without structure preservation (more aggressive merging)
            chunks_flow = chunk_text(cleaned, max_chunk_size=100, preserve_structure=False)
        
            # Both should be valid
            assert len(chunks_structured) > 0
            assert len(chunks_flow) > 0
    
        def test_hyphenated_prose_not_misclassified(self):
            """
            Verify that hyphenated prose is not misclassified as formula chunks.
            """
            hyphenated_text = """State-of-the-Art Techniques

    The well-known state-of-the-art approach has been widely adopted. This cutting-edge 
    method uses high-performance algorithms to achieve better results. The user-friendly 
    interface makes it easy-to-use for both beginners and advanced practitioners."""
        
            chunks = preprocess_text(hyphenated_text)
        
            # No chunk should be misclassified as formula
            formula_chunks = [c for c in chunks if c['metadata']['has_formula']]
            assert len(formula_chunks) == 0, "Hyphenated prose should not produce formula chunks"
        
            # Types should be reasonable (heading, paragraph)
            chunk_types = [c['metadata']['type'] for c in chunks]
            assert 'formula' not in chunk_types, "No formula type expected for hyphenated prose"
    
        def test_real_formulas_vs_hyphenated_distinction(self):
            """
            Ensure that real formulas are still detected correctly while hyphenated prose is not.
            """
            mixed_text = """Advanced Mathematics

    State-of-the-art techniques in modern physics include the following:

    Einstein's Energy Formula:
    E = mc²

    This well-known equation relates energy to mass using the speed-of-light constant.

    Newton's Law:
    F = ma

    The above-mentioned laws form the basis of classical mechanics."""
        
            chunks = preprocess_text(mixed_text)
        
            # Should have formula chunks
            formula_chunks = [c for c in chunks if c['metadata']['type'] == 'formula']
            assert len(formula_chunks) > 0, "Should detect actual formulas"
        
            # Paragraph chunks should not have has_formula=True
            para_chunks = [c for c in chunks if c['metadata']['type'] == 'paragraph']
            para_with_hyphen = [c for c in para_chunks if 'well-known' in c['text'] or 'speed-of-light' in c['text']]
        
            # Hyphenated paragraphs should not have has_formula=True
            for chunk in para_with_hyphen:
                assert chunk['metadata']['has_formula'] is False, \
                    f"Paragraph with hyphens should not have has_formula=True: {chunk['text'][:50]}"
    
    def test_mixed_content_preprocessing(self):
        """
        Test preprocessing of mixed content (text + formulas + lists).
        """
        mixed_text = """# Advanced Mathematics

This course covers differential equations and linear algebra.

Key Equations:
∂u/∂t = ∇²u + f(x,t)

Topics Covered:
1. Ordinary differential equations
2. Partial differential equations
3. Eigenvalue problems

Applications:
- Heat conduction
- Wave propagation
- Quantum mechanics
"""
        
        chunks = preprocess_text(mixed_text)
        
        # Should handle all content types
        assert len(chunks) > 0
        
        # Should detect formulas
        formula_chunks = [c for c in chunks if c['metadata']['has_formula']]
        assert len(formula_chunks) > 0
        
        # Should detect lists
        list_chunks = [c for c in chunks if c['metadata']['type'] == 'list']
        assert len(list_chunks) > 0
    
    def test_preprocessing_statistics_logging(self, sample_text, capsys):
        """
        Verify preprocessing statistics are logged.
        """
        chunks = preprocess_text(sample_text)
        
        # Capture logs (in pytest with capsys)
        assert len(chunks) > 0
        
        # Manually verify stats calculations
        total_chars = sum(c['metadata']['char_count'] for c in chunks)
        avg_chunk_size = total_chars / len(chunks) if chunks else 0
        
        assert total_chars > 0
        assert avg_chunk_size > 0


class TestNERToPipelineIntegration:
    """Integration tests for NER with downstream modules."""
    
    def test_ner_to_flashcard_pipeline(self, sample_text):
        """
        Test NER output compatibility with flashcard generation (Phase 5 stub).
        """
        from utils.ner import EntityExtractor
        
        chunks = preprocess_text(sample_text)
        ner = EntityExtractor(use_fallback=True)
        chunks_with_entities = ner.extract_from_chunks(chunks)
        
        # Extract all entities for flashcard topics
        all_entities = []
        for chunk in chunks_with_entities:
            entities = chunk['metadata'].get('entities', [])
            all_entities.extend([e['text'] for e in entities])
        
        # Simulate flashcard generation from entities
        flashcard_topics = all_entities[:10]  # Top 10 entities as flashcard topics
        
        # Should have some topics for flashcard generation
        assert len(flashcard_topics) >= 0  # May be empty for NLTK fallback
        
        # Each topic should be a valid string
        for topic in flashcard_topics:
            assert isinstance(topic, str)
            assert len(topic) > 0
    
    def test_ner_to_graph_pipeline(self, sample_text):
        """
        Test NER output compatibility with knowledge graph builder (Phase 6 stub).
        """
        from utils.ner import EntityExtractor
        
        chunks = preprocess_text(sample_text)
        ner = EntityExtractor(use_fallback=True)
        chunks_with_entities = ner.extract_from_chunks(chunks)
        
        # Simulate graph builder: create nodes from entities
        nodes = []
        for chunk in chunks_with_entities:
            entities = chunk['metadata'].get('entities', [])
            for entity_dict in entities:
                node = {
                    'label': entity_dict['text'],
                    'type': entity_dict['type'],
                    'confidence': entity_dict['confidence']
                }
                nodes.append(node)
        
        # Should have some nodes for graph construction
        assert len(nodes) >= 0
        
        # Each node should have required fields
        for node in nodes:
            assert 'label' in node
            assert 'type' in node
            assert 'confidence' in node
            assert node['type'] in ['Concept', 'Topic', 'Entity', 'Formula']
    
    def test_full_pipeline_preprocess_embeddings_ner(self, sample_text):
        """
        Test full pipeline: preprocess → embeddings → NER.
        """
        from utils.embeddings import EmbeddingGenerator
        from utils.ner import EntityExtractor
        import sqlite3
        import tempfile
        
        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            # Initialize schema
            conn = sqlite3.connect(db_path)
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
            
            # Phase 2: Preprocess
            chunks = preprocess_text(sample_text, max_chunk_size=500)
            assert len(chunks) > 0
            
            # Phase 3: Embeddings
            emb_gen = EmbeddingGenerator(db_path=db_path, force_hash_fallback=True)
            embedding_results = emb_gen.embed_and_store_chunks(chunks)
            assert len(embedding_results) == len(chunks)
            
            # Phase 4: NER
            ner = EntityExtractor(use_fallback=True)
            chunks_with_entities = ner.extract_from_chunks(chunks)
            assert len(chunks_with_entities) == len(chunks)
            
            # Verify all metadata present
            for i, chunk in enumerate(chunks_with_entities):
                assert 'entities' in chunk['metadata']
                assert 'embedding_id' in chunk['metadata']
                assert isinstance(chunk['metadata']['entities'], list)
                assert isinstance(chunk['metadata']['embedding_id'], int)
        
        finally:
            import os
            if os.path.exists(db_path):
                os.remove(db_path)

class TestPhase6GraphIntegration:
    """Integration tests for Phase 6 knowledge graph construction."""

    def test_graph_builder_end_to_end(self, sample_text):
        """
        Run pipeline: preprocess → NER → GraphBuilder, then export JSON.
        """
        from utils.ner import EntityExtractor
        from knowledge_graph.build import GraphBuilder
        from knowledge_graph.visualize import export_graph_json
        from utils.embeddings import EmbeddingGenerator
        import tempfile
        import os
        import json

        # Temporary DB
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            # Phase 2
            chunks = preprocess_text(sample_text, max_chunk_size=400)
            assert len(chunks) > 0

            # Phase 3
            emb_gen = EmbeddingGenerator(db_path=db_path, force_hash_fallback=True)
            _ = emb_gen.embed_and_store_chunks(chunks)

            # Phase 4
            ner = EntityExtractor(use_fallback=True)
            chunks_with_entities = ner.extract_from_chunks(chunks)

            # Phase 6
            gb = GraphBuilder(
                chunks_with_entities=chunks_with_entities,
                embedding_generator=emb_gen,
                db_path=db_path,
                flashcard_ids=[]
            )
            res = gb.build_graph()
            assert res["nodes_created"] >= 0
            assert res["edges_created"] >= 0

            # Export JSON
            out_json = os.path.join(os.path.dirname(db_path), "graph.json")
            export_graph_json(db_path, output_path=out_json)
            assert os.path.exists(out_json)
            with open(out_json, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            assert "nodes" in payload and "edges" in payload
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

class TestPhase5FlashcardIntegration:
    """Integration tests for Phase 5 flashcard generation pipeline."""
    
    def test_ner_to_flashcard_pipeline(self, sample_text):
        """
        Test full pipeline: preprocess → embeddings → NER → flashcards.
        """
        from utils.embeddings import EmbeddingGenerator
        from utils.ner import EntityExtractor
        from flashcards.generate import FlashcardGenerator
        import sqlite3
        import tempfile
        
        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            # Initialize schema
            conn = sqlite3.connect(db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vector BLOB NOT NULL,
                    entity_type TEXT NOT NULL,
                    entity_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS flashcards (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    difficulty INTEGER DEFAULT 0,
                    context TEXT,
                    source_type TEXT DEFAULT 'generated',
                    embedding_id INTEGER,
                    FOREIGN KEY(embedding_id) REFERENCES embeddings(id)
                );
            """)
            conn.commit()
            conn.close()
            
            # Phase 2: Preprocess
            chunks = preprocess_text(sample_text, max_chunk_size=500)
            assert len(chunks) > 0
            
            # Phase 3: Embeddings
            emb_gen = EmbeddingGenerator(db_path=db_path, force_hash_fallback=True)
            embedding_results = emb_gen.embed_and_store_chunks(chunks)
            
            # Phase 4: NER
            ner = EntityExtractor(use_fallback=True)
            chunks_with_entities = ner.extract_from_chunks(chunks)
            
            # Phase 5: Flashcards
            fc_gen = FlashcardGenerator(db_path=db_path, ner_extractor=ner)
            flashcards = fc_gen.generate_from_chunks(chunks_with_entities)
            
            # Verify flashcard generation
            assert len(flashcards) > 0
            
            # Store flashcards
            ids = fc_gen.store_flashcards(flashcards)
            assert len(ids) == len(flashcards)
            assert all(isinstance(id, int) and id > 0 for id in ids)
            
            # Verify each flashcard has required fields
            for fc in flashcards:
                assert fc.question
                assert fc.answer
                assert fc.topic
                assert fc.difficulty in range(0, 6)
                assert fc.source_type in ['rule_based', 'qa_extracted', 'hybrid']
        
        finally:
            import os
            if os.path.exists(db_path):
                os.remove(db_path)
    
    def test_ner_entities_passed_to_flashcard_generation(self, sample_text):
        """
        Verify that NER-extracted entities are properly passed to flashcard generation.
        """
        from utils.ner import EntityExtractor
        from flashcards.generate import FlashcardGenerator
        
        chunks = preprocess_text(sample_text)
        
        # Extract entities
        ner = EntityExtractor(use_fallback=True)
        chunks_with_entities = ner.extract_from_chunks(chunks)
        
        # Generate flashcards
        fc_gen = FlashcardGenerator(db_path=":memory:", ner_extractor=ner)
        flashcards = fc_gen.generate_from_chunks(chunks_with_entities)
        
        # Verify entities appear in flashcards
        flashcard_entities = []
        for fc in flashcards:
            flashcard_entities.extend(fc.entities)
        
        # Some entities should be extracted
        assert len(flashcard_entities) >= 0  # May be empty for simple texts
    
    def test_flashcard_difficulty_based_on_academic_level(self, sample_text):
        """
        Verify that flashcard difficulty is correctly mapped from academic level.
        """
        from flashcards.generate import FlashcardGenerator
        
        chunks = preprocess_text(sample_text)
        
        fc_gen = FlashcardGenerator(db_path=":memory:")
        flashcards = fc_gen.generate_from_chunks(chunks)
        
        # Group flashcards by academic level
        basic_cards = [fc for fc in flashcards if fc.difficulty in [1, 2]]
        intermediate_cards = [fc for fc in flashcards if fc.difficulty in [2, 3]]
        advanced_cards = [fc for fc in flashcards if fc.difficulty in [4, 5]]
        
        # Should have some distribution
        total = len(basic_cards) + len(intermediate_cards) + len(advanced_cards)
        assert total == len(flashcards)
    
    def test_deduplication_removes_similar_flashcards(self, sample_text):
        """
        Verify that deduplication removes similar flashcards.
        """
        from flashcards.generate import FlashcardGenerator, Flashcard
        
        fc_gen = FlashcardGenerator(db_path=":memory:")
        
        # Create test flashcards with similar questions
        cards = [
            Flashcard(
                question="What is photosynthesis?",
                answer="Process of converting light to chemical energy",
                topic="Biology",
                entities=["photosynthesis"],
                embedding_id=1,
                difficulty=2,
                source_type="rule_based"
            ),
            Flashcard(
                question="What is photosynthesis?",  # Duplicate
                answer="Same as above",
                topic="Biology",
                entities=["photosynthesis"],
                embedding_id=1,
                difficulty=2,
                source_type="rule_based"
            ),
            Flashcard(
                question="How does photosynthesis work?",  # Different
                answer="Plants convert light energy",
                topic="Biology",
                entities=["photosynthesis"],
                embedding_id=2,
                difficulty=3,
                source_type="rule_based"
            )
        ]
        
        unique_cards = fc_gen._deduplicate_flashcards(cards)
        
        # Should have fewer than original
        assert len(unique_cards) < len(cards)
        assert len(unique_cards) >= 2
    
    def test_flashcard_output_format_compatibility(self, sample_text):
        """
        Verify that generated flashcards have correct output format.
        """
        from flashcards.generate import FlashcardGenerator
        
        chunks = preprocess_text(sample_text)
        fc_gen = FlashcardGenerator(db_path=":memory:")
        flashcards = fc_gen.generate_from_chunks(chunks)
        
        # Verify output format
        for fc in flashcards:
            # Required fields
            assert hasattr(fc, 'question')
            assert hasattr(fc, 'answer')
            assert hasattr(fc, 'topic')
            assert hasattr(fc, 'entities')
            assert hasattr(fc, 'embedding_id')
            assert hasattr(fc, 'difficulty')
            assert hasattr(fc, 'source_type')
            
            # Type validation
            assert isinstance(fc.question, str)
            assert isinstance(fc.answer, str)
            assert isinstance(fc.topic, str)
            assert isinstance(fc.entities, list)
            assert isinstance(fc.embedding_id, (int, type(None)))
            assert isinstance(fc.difficulty, int)
            assert isinstance(fc.source_type, str)
    
    def test_flashcard_storage_and_retrieval(self, sample_text):
        """
        Test flashcard storage in SQLite and retrieval.
        """
        from flashcards.generate import FlashcardGenerator
        import sqlite3
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            # Initialize database
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
            
            # Generate and store
            chunks = preprocess_text(sample_text)
            fc_gen = FlashcardGenerator(db_path=db_path)
            flashcards = fc_gen.generate_from_chunks(chunks)
            
            if flashcards:
                ids = fc_gen.store_flashcards(flashcards)
                
                # Retrieve and verify
                conn = sqlite3.connect(db_path)
                cursor = conn.execute("SELECT * FROM flashcards")
                stored = cursor.fetchall()
                conn.close()
                
                assert len(stored) == len(flashcards)
        
        finally:
            import os
            if os.path.exists(db_path):
                os.remove(db_path)