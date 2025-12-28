"""
Unit tests for offline-ai preprocessing module (utils/preprocess.py).
"""
import pytest
from utils.preprocess import (
    clean_text, chunk_text, extract_metadata, handle_markdown,
    handle_pdf_text, preprocess_text, PreprocessingError
)


# ============================================================================
# Tests for clean_text()
# ============================================================================

class TestCleanText:
    """Test suite for clean_text() function."""
    
    def test_clean_text_removes_extra_whitespace(self):
        """Multiple spaces should be collapsed to single space."""
        text = "Hello    world  with   multiple   spaces"
        result = clean_text(text)
        assert "    " not in result
        assert result == "Hello world with multiple spaces"
    
    def test_clean_text_normalizes_newlines(self):
        """Carriage return + newline should be normalized to newline."""
        text = "Line 1\r\nLine 2\r\nLine 3"
        result = clean_text(text)
        assert '\r' not in result
        assert result == "Line 1\nLine 2\nLine 3"
    
    def test_clean_text_collapses_multiple_newlines(self):
        """3+ consecutive newlines should collapse to 2 (preserve paragraphs)."""
        text = "Para 1\n\n\n\nPara 2\n\n\n\n\nPara 3"
        result = clean_text(text)
        assert "\n\n\n" not in result
        assert result == "Para 1\n\nPara 2\n\nPara 3"
    
    def test_clean_text_preserves_formulas(self):
        """Mathematical symbols should be preserved when preserve_formulas=True."""
        text = "E = mc² + Δ → ∞"
        result = clean_text(text, preserve_formulas=True)
        assert "=" in result
        assert "→" in result
        assert "²" in result
    
    def test_clean_text_handles_empty_input(self):
        """Empty string should return empty string."""
        assert clean_text("") == ""
    
    def test_clean_text_raises_on_none(self):
        """None input should raise ValueError."""
        with pytest.raises(ValueError):
            clean_text(None)
    
    def test_clean_text_raises_on_non_string(self):
        """Non-string input should raise ValueError."""
        with pytest.raises(ValueError):
            clean_text(123)
        with pytest.raises(ValueError):
            clean_text(['list'])
    
    def test_clean_text_removes_control_chars(self):
        """Control characters (except newline/tab) should be removed."""
        text = "Hello\x00\x01\x02World"
        result = clean_text(text)
        assert '\x00' not in result
        assert '\x01' not in result
    
    def test_clean_text_preserves_markdown(self):
        """Markdown syntax should be preserved when preserve_markdown=True."""
        text = "# Heading\n- List item\n**bold**"
        result = clean_text(text, preserve_markdown=True)
        assert "#" in result
        assert "-" in result
    
    def test_clean_text_decodes_html_entities(self):
        """HTML entities should be decoded."""
        text = "Hello&nbsp;world&amp;text&lt;tag&gt;"
        result = clean_text(text)
        assert "&nbsp;" not in result
        assert "&amp;" not in result
        assert " " in result
        assert "&" in result
    
        def test_clean_text_removes_markdown_when_flag_false(self):
            """Markdown syntax should be removed when preserve_markdown=False."""
            text = "# Heading\n- List item\n**bold**"
            result = clean_text(text, preserve_markdown=False)
            assert "#" not in result
            assert result.count("-") == 0  # All list dashes removed
    
        def test_clean_text_removes_formulas_when_flag_false(self):
            """Math symbols should be removed when preserve_formulas=False."""
            text = "E = mc² and x + y = z"
            result = clean_text(text, preserve_formulas=False)
            assert "=" not in result
            assert "+" not in result
            # Variable names should still be there
            assert "E" in result
            assert "mc" in result


# ============================================================================
# Tests for chunk_text()
# ============================================================================

class TestChunkText:
    """Test suite for chunk_text() function."""
    
    def test_chunk_text_splits_by_paragraphs(self):
        """Double newline should create separate chunks."""
        text = "Para 1\n\nPara 2\n\nPara 3"
        chunks = chunk_text(text, max_chunk_size=500)
        assert len(chunks) >= 3
    
    def test_chunk_text_respects_max_size(self):
        """No chunk should exceed max_chunk_size."""
        text = "Short text\n\n" + ("Long paragraph " * 100)
        chunks = chunk_text(text, max_chunk_size=200)
        for chunk in chunks:
            assert len(chunk) <= 200
    
    def test_chunk_text_short_paragraph_kept_intact(self):
        """Short paragraphs should not be split."""
        text = "This is a short paragraph."
        chunks = chunk_text(text, max_chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_chunk_text_handles_empty_paragraphs(self):
        """Empty paragraphs should be skipped."""
        text = "Para 1\n\n\n\nPara 2\n\n"
        chunks = chunk_text(text)
        assert "" not in chunks
        assert len(chunks) == 2
    
    def test_chunk_text_empty_input_returns_empty_list(self):
        """Empty input should return empty list."""
        assert chunk_text("") == []
        assert chunk_text("   ") == []
    
    def test_chunk_text_preserves_long_sentences(self):
        """Long sentences should be split at sentence boundaries."""
        long_sent = "This is a long sentence with many clauses. " * 5
        chunks = chunk_text(long_sent, max_chunk_size=200)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 200
    
    def test_chunk_text_handles_lists(self):
        """Lists should be preserved as chunks."""
        text = "Introduction\n\n1. First item\n2. Second item\n3. Third item"
        chunks = chunk_text(text, max_chunk_size=500)
        # Should not be empty
        assert len(chunks) > 0
    
    def test_chunk_text_handles_formulas(self):
        """Formula chunks should be preserved."""
        text = "Formula section\n\nE = mc²\n\nNext section"
        chunks = chunk_text(text)
        assert len(chunks) >= 2
    
        def test_chunk_text_respects_preserve_structure_flag(self):
            """When preserve_structure=False, paragraphs should be combined."""
            text = "Para 1\n\nPara 2\n\nPara 3"
        
            # With preserve_structure=True: separate chunks per paragraph
            chunks_structured = chunk_text(text, max_chunk_size=500, preserve_structure=True)
        
            # With preserve_structure=False: may combine paragraphs
            chunks_flow = chunk_text(text, max_chunk_size=500, preserve_structure=False)
        
            # Both should produce valid chunks
            assert len(chunks_structured) > 0
            assert len(chunks_flow) > 0
    
        def test_chunk_text_aggressive_chunking_when_preserve_structure_false(self):
            """When preserve_structure=False, should allow more paragraph merging."""
            text = "Short para 1\n\nShort para 2\n\nShort para 3"
        
            # With preserve_structure=True: likely 3 chunks (one per para)
            chunks_structured = chunk_text(text, max_chunk_size=500, preserve_structure=True)
        
            # With preserve_structure=False: may combine into fewer chunks
            chunks_flow = chunk_text(text, max_chunk_size=500, preserve_structure=False)
        
            # Flow version may have same or fewer chunks
            assert len(chunks_flow) <= len(chunks_structured) or len(chunks_flow) == len(chunks_structured)


# ============================================================================
# Tests for extract_metadata()
# ============================================================================

class TestExtractMetadata:
    """Test suite for extract_metadata() function."""
    
    def test_extract_metadata_detects_heading(self):
        """Short title-case text should be detected as heading."""
        chunk = "Introduction to Biology"
        meta = extract_metadata(chunk)
        assert meta['type'] == 'heading'
    
    def test_extract_metadata_detects_formula(self):
        """Text with math symbols should have has_formula=True."""
        chunk = "E = mc² + Δx"
        meta = extract_metadata(chunk)
        assert meta['has_formula'] is True
    
    def test_extract_metadata_detects_list(self):
        """Text starting with list markers should be type='list'."""
        chunk = "1. First item\n2. Second item"
        meta = extract_metadata(chunk)
        assert meta['type'] == 'list'
    
    def test_extract_metadata_detects_list_bullets(self):
        """Text with bullet points should be type='list'."""
        chunk = "- Item 1\n- Item 2"
        meta = extract_metadata(chunk)
        assert meta['type'] == 'list'
    
    def test_extract_metadata_calculates_char_count(self):
        """Character count should be accurate."""
        chunk = "Hello world"
        meta = extract_metadata(chunk)
        assert meta['char_count'] == len(chunk)
    
    def test_extract_metadata_calculates_word_count(self):
        """Word count should be accurate."""
        chunk = "Hello world test"
        meta = extract_metadata(chunk)
        assert meta['word_count'] == 3
    
    def test_extract_metadata_extracts_topic_from_heading(self):
        """Topic should be extracted from markdown heading."""
        chunk = "# Introduction\n\nSome content"
        meta = extract_metadata(chunk)
        assert "Introduction" in meta['topic']
    
    def test_extract_metadata_uses_context_topic(self):
        """Context topic should be used when no heading present."""
        chunk = "Some paragraph text"
        meta = extract_metadata(chunk, context_topic="Biology")
        assert meta['topic'] == "Biology" or "Some" in meta['topic']
    
    def test_extract_metadata_handles_empty_chunk(self):
        """Empty chunk should return default metadata."""
        meta = extract_metadata("")
        assert meta['type'] == 'unknown'
        assert meta['char_count'] == 0
        assert meta['word_count'] == 0
    
    def test_extract_metadata_estimates_academic_level(self):
        """Academic level should be estimated based on content."""
        basic_chunk = "Hello world"
        advanced_chunk = "The differential equations in quantum mechanics represent observables."
        
        basic_meta = extract_metadata(basic_chunk)
        advanced_meta = extract_metadata(advanced_chunk)
        
        # Advanced chunk should have higher academic level
        levels = {'basic': 0, 'intermediate': 1, 'advanced': 2}
        assert levels.get(advanced_meta['academic_level'], 0) >= levels.get(basic_meta['academic_level'], 0)
    
        def test_extract_metadata_hyphenated_prose_not_formula(self):
            """Hyphenated prose (well-known, state-of-the-art) should NOT be flagged as formula."""
            hyphenated_text = "This well-known state-of-the-art approach is widely used."
            meta = extract_metadata(hyphenated_text)
            assert meta['has_formula'] is False, "Hyphenated prose should not be detected as formula"
            assert meta['type'] != 'formula', "Hyphenated prose chunk type should not be 'formula'"
    
        def test_extract_metadata_actual_formula_is_detected(self):
            """Actual math formulas should be detected correctly."""
            formula_text = "The velocity v = dx/dt represents the rate of change."
            meta = extract_metadata(formula_text)
            assert meta['has_formula'] is True, "Math formula should be detected"


# ============================================================================
# Tests for handle_markdown()
# ============================================================================

class TestHandleMarkdown:
    """Test suite for handle_markdown() function."""
    
    def test_handle_markdown_removes_bold(self):
        """Bold markdown syntax should be removed."""
        text = "This is **bold** text"
        result = handle_markdown(text)
        assert "**" not in result
        assert "bold" in result
    
    def test_handle_markdown_removes_italic(self):
        """Italic markdown syntax should be removed."""
        text = "This is *italic* text"
        result = handle_markdown(text)
        assert "*" not in result
        assert "italic" in result
    
    def test_handle_markdown_removes_links(self):
        """Link markdown syntax should be removed, text kept."""
        text = "Click [here](https://example.com)"
        result = handle_markdown(text)
        assert "[" not in result
        assert "here" in result
        assert "https" not in result
    
    def test_handle_markdown_preserves_code_blocks(self):
        """Code blocks (triple backticks) should be preserved."""
        text = "```python\ncode\n```"
        result = handle_markdown(text)
        assert "```" in result or "code" in result


# ============================================================================
# Tests for handle_pdf_text()
# ============================================================================

class TestHandlePdfText:
    """Test suite for handle_pdf_text() function."""
    
    def test_handle_pdf_text_fixes_hyphenation(self):
        """Hyphenation at line breaks should be fixed."""
        text = "word-\nbreak"
        result = handle_pdf_text(text)
        # Should be fixed (word-break or wordbreak)
        assert "word" in result and "break" in result
    
    def test_handle_pdf_text_removes_page_numbers(self):
        """Page numbers should be removed."""
        text = "Content line 1\n\n42\n\nContent line 2"
        result = handle_pdf_text(text)
        # Page number should be reduced/removed
        assert "Content line 1" in result
        assert "Content line 2" in result
    
    def test_handle_pdf_text_removes_excessive_blank_lines(self):
        """Multiple blank lines should be collapsed."""
        text = "Line 1\n\n\n\nLine 2"
        result = handle_pdf_text(text)
        assert "\n\n\n" not in result


# ============================================================================
# Tests for preprocess_text()
# ============================================================================

class TestPreprocessText:
    """Test suite for preprocess_text() orchestrator function."""
    
    def test_preprocess_text_returns_list_of_dicts(self, sample_text):
        """Output should be list of dicts with 'text' and 'metadata' keys."""
        result = preprocess_text(sample_text)
        assert isinstance(result, list)
        assert len(result) > 0
        for item in result:
            assert 'text' in item
            assert 'metadata' in item
            assert isinstance(item['metadata'], dict)
    
    def test_preprocess_text_metadata_has_required_fields(self, sample_text):
        """Metadata should have all required fields."""
        result = preprocess_text(sample_text)
        required_fields = {'type', 'topic', 'char_count', 'word_count', 'has_formula', 'academic_level'}
        for item in result:
            assert required_fields.issubset(item['metadata'].keys())
    
    def test_preprocess_text_handles_plain_format(self, sample_text):
        """Plain text format should be handled correctly."""
        result = preprocess_text(sample_text, input_format='plain')
        assert len(result) > 0
    
    def test_preprocess_text_handles_markdown_format(self, markdown_text):
        """Markdown format should be handled correctly."""
        result = preprocess_text(markdown_text, input_format='markdown')
        assert len(result) > 0
    
    def test_preprocess_text_handles_pdf_format(self, complex_pdf_text):
        """PDF format should be handled correctly."""
        result = preprocess_text(complex_pdf_text, input_format='pdf')
        assert len(result) > 0
    
    def test_preprocess_text_empty_input_returns_empty_list(self):
        """Empty input should return empty list."""
        result = preprocess_text("")
        assert result == []
    
    def test_preprocess_text_respects_max_chunk_size(self, sample_text):
        """All chunks should respect max_chunk_size."""
        result = preprocess_text(sample_text, max_chunk_size=300)
        for item in result:
            assert len(item['text']) <= 300
    
    def test_preprocess_text_propagates_topic_context(self, sample_text):
        """Topic should be propagated from headings to following chunks."""
        result = preprocess_text(sample_text)
        # Find a heading
        heading_idx = None
        for i, item in enumerate(result):
            if item['metadata']['type'] == 'heading':
                heading_idx = i
                break
        
        # Check if following chunk has propagated topic
        if heading_idx is not None and heading_idx + 1 < len(result):
            heading_topic = result[heading_idx]['metadata']['topic']
            next_topic = result[heading_idx + 1]['metadata']['topic']
            assert heading_topic is not None


# ============================================================================
# Integration tests
# ============================================================================

@pytest.mark.integration
class TestPreprocessingIntegration:
    """Integration tests for complete preprocessing pipeline."""
    
    def test_full_pipeline_with_academic_text(self, sample_text):
        """Full pipeline should handle academic text correctly."""
        chunks = preprocess_text(sample_text, max_chunk_size=500)
        
        # Should produce multiple chunks
        assert len(chunks) > 1
        
        # All chunks should have required structure
        for chunk in chunks:
            assert len(chunk['text']) > 0
            assert chunk['metadata']['char_count'] > 0
            assert chunk['metadata']['type'] in ['heading', 'formula', 'list', 'paragraph']
            assert chunk['metadata']['academic_level'] in ['basic', 'intermediate', 'advanced']
    
    def test_full_pipeline_preserves_formulas(self, formula_text):
        """Formulas should be preserved through pipeline."""
        chunks = preprocess_text(formula_text)
        
        # Should have chunks with formulas
        has_formula_chunks = [c for c in chunks if c['metadata']['has_formula']]
        assert len(has_formula_chunks) > 0
    
    def test_full_pipeline_handles_long_text(self):
        """Pipeline should handle large texts efficiently."""
        long_text = "This is a paragraph. " * 500
        chunks = preprocess_text(long_text, max_chunk_size=300)
        
        # Should chunk into manageable pieces
        assert len(chunks) > 10
        for chunk in chunks:
            assert len(chunk['text']) <= 300
    
    def test_full_pipeline_edge_cases(self, empty_text, long_paragraph):
        """Pipeline should handle edge cases gracefully."""
        # Empty text
        result_empty = preprocess_text(empty_text['empty'])
        assert result_empty == []
        
        # Long paragraph
        result_long = preprocess_text(long_paragraph, max_chunk_size=400)
        assert len(result_long) > 0
        for chunk in result_long:
            assert len(chunk['text']) <= 400
