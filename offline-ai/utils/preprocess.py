"""
Text preprocessing and chunking utilities for offline AI system.

This module provides utilities for cleaning, normalizing, and chunking academic text
while preserving structural elements (headings, lists, formulas). It serves as Phase 2
of the offline AI pipeline, preparing text for downstream modules (embeddings, NER,
flashcard generation).

Key Features:
- Text cleaning: Remove noise, normalize whitespace, preserve special characters
- Semantic chunking: Split text by paragraphs and sentences (NLTK)
- Metadata extraction: Detect content type, topic, formulas, academic level
- Format handling: Support for plain text, markdown, and PDF-extracted text
- Error handling: Graceful handling of edge cases and malformed input

Dependencies:
- NLTK (nltk.tokenizers.punkt for sentence tokenization)
- regex (standard library)

Usage Example:
    from utils.preprocess import preprocess_text
    
    text = open('academic_paper.txt').read()
    chunks = preprocess_text(text, max_chunk_size=500)
    
    for chunk in chunks:
        print(f"Type: {chunk['metadata']['type']}")
        print(f"Topic: {chunk['metadata']['topic']}")
        print(f"Text: {chunk['text'][:100]}...")
"""

import logging
import re
import nltk
from typing import List, Dict, Any, Optional
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Download NLTK punkt data on first run
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)


class PreprocessingError(Exception):
    """Custom exception for preprocessing errors."""
    pass


def clean_text(
    text: str,
    preserve_formulas: bool = True,
    preserve_markdown: bool = True
) -> str:
    """
    Clean and normalize text by removing noise and standardizing whitespace.
    
    This function:
    - Removes excessive whitespace (multiple spaces/tabs/newlines)
    - Normalizes line endings (\r\n → \n)
    - Collapses 3+ newlines to 2 (preserves paragraph breaks)
    - Strips leading/trailing whitespace from each line
    - Optionally preserves formulas and markdown syntax
    - Removes control characters (except newlines/tabs)
    - Decodes HTML entities if present
    
    Args:
        text (str): Input text to clean. Must be a string.
        preserve_formulas (bool): If True, preserve mathematical symbols (=, +, →, etc).
                                 Default: True
        preserve_markdown (bool): If True, preserve markdown syntax (#, -, *, etc).
                                 Default: True
    
    Returns:
        str: Cleaned and normalized text.
    
    Raises:
        ValueError: If text is None or not a string.
    
    Examples:
        >>> clean_text("Hello  world\\n\\n\\n  text")
        'Hello world\\n\\ntext'
        
        >>> clean_text("E = mc²", preserve_formulas=True)
        'E = mc²'
    """
    if text is None or not isinstance(text, str):
        raise ValueError("Input text must be a string, not None or other type")
    
    try:
        # Normalize line endings: \r\n → \n
        text = text.replace('\r\n', '\n')
        
        # Decode HTML entities if present
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        
        # Remove control characters (except newline, tab)
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        
        # Strip leading/trailing whitespace from each line
        lines = text.split('\n')
        lines = [line.rstrip() for line in lines]
        text = '\n'.join(lines)
        
        # Collapse multiple spaces to single space (preserve initial indentation)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Collapse 3+ newlines to 2 (preserve paragraph breaks)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove trailing whitespace per line
        text = '\n'.join(line.rstrip() for line in text.split('\n'))
        
        logger.debug(f"Text cleaned: {len(text)} characters")
        return text
    
    except Exception as e:
        logger.error(f"Error cleaning text: {e}")
        raise PreprocessingError(f"Failed to clean text: {e}")


def chunk_text(
        raise ValueError("Input text must be a string, not None or other type")
    
    try:
        # Normalize line endings: \r\n → \n
        text = text.replace('\r\n', '\n')
        
        # Decode HTML entities if present
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        
        # Remove control characters (except newline, tab)
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        
        # Strip leading/trailing whitespace from each line
        lines = text.split('\n')
        lines = [line.rstrip() for line in lines]
        text = '\n'.join(lines)
        
        # Collapse multiple spaces to single space (preserve initial indentation)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Collapse 3+ newlines to 2 (preserve paragraph breaks)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove trailing whitespace per line
        text = '\n'.join(line.rstrip() for line in text.split('\n'))
        
        # Remove markdown markers if preserve_markdown is False
        if not preserve_markdown:
            text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)  # Remove markdown headings
            text = re.sub(r'^\s*[-*•]\s*', '', text, flags=re.MULTILINE)  # Remove bullet points
        
        # Remove math symbols if preserve_formulas is False
        if not preserve_formulas:
            text = re.sub(r'[=+×÷→⇒/]', '', text)  # Remove strong math operators
            text = re.sub(r'[₀-₉₊₋₌⁰-⁹⁺⁻⁼]', '', text)  # Remove subscripts/superscripts
            text = re.sub(r'\$[^\$]*\$', '', text)  # Remove LaTeX math
        
        logger.debug(f"Text cleaned: {len(text)} characters (preserve_formulas={preserve_formulas}, preserve_markdown={preserve_markdown})")
        return text
    
    except Exception as e:
        logger.error(f"Error cleaning text: {e}")
        raise PreprocessingError(f"Failed to clean text: {e}")
        # Split into paragraphs
        paragraphs = text.split('\n\n')
        chunks = []
        
        for para_idx, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            
            # Skip empty paragraphs
            if not paragraph:
                continue
            
            # If paragraph is short enough, add as single chunk
            if len(paragraph) <= max_chunk_size:
                chunks.append(paragraph)
                continue
            
            # For long paragraphs, split into sentences
            try:
                sentences = nltk.sent_tokenize(paragraph)
            except Exception as e:
                logger.warning(f"NLTK tokenization failed: {e}. Splitting by period.")
                sentences = [s.strip() + '.' for s in paragraph.split('. ')]
            
            # Group sentences into chunks
            current_chunk = ""
            for sent_idx, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # Check if adding this sentence would exceed max_chunk_size
                test_chunk = current_chunk + (" " if current_chunk else "") + sentence
                
                if len(test_chunk) <= max_chunk_size:
                    current_chunk = test_chunk
                else:
                    # Current chunk is full, save it and start new
                    if current_chunk:
                        chunks.append(current_chunk)
                    
                    # If single sentence is too long, split at commas/semicolons
                    if len(sentence) > max_chunk_size:
                        subsents = _split_long_sentence(sentence, max_chunk_size)
                        chunks.extend(subsents)
                        current_chunk = ""
                    else:
                        current_chunk = sentence
            
            # Add remaining chunk
            if current_chunk:
                chunks.append(current_chunk)
        
        logger.info(f"Text chunked into {len(chunks)} chunks (max_size={max_chunk_size})")
        return chunks
    
                logger.warning("Empty or invalid text for chunking")
                return []
    
            try:
                # When preserve_structure=False, combine all paragraphs for more aggressive chunking
                if not preserve_structure:
                    # Replace paragraph breaks with spaces to allow free-flow chunking
                    text = text.replace('\n\n', ' ')
                    paragraphs = [text]
                else:
                    # Standard: split by paragraph breaks
                    paragraphs = text.split('\n\n')
        
                chunks = []
        
                for para_idx, paragraph in enumerate(paragraphs):
                    paragraph = paragraph.strip()
            
                    # Skip empty paragraphs
                    if not paragraph:
                        continue
            
                    # If paragraph is short enough, add as single chunk
                    if len(paragraph) <= max_chunk_size:
                        chunks.append(paragraph)
                        continue
            
                    # For long paragraphs, split into sentences
                    try:
                        sentences = nltk.sent_tokenize(paragraph)
                    except Exception as e:
                        logger.warning(f"NLTK tokenization failed: {e}. Splitting by period.")
                        sentences = [s.strip() + '.' for s in paragraph.split('. ')]
            
                    # Group sentences into chunks
                    current_chunk = ""
                    for sent_idx, sentence in enumerate(sentences):
                        sentence = sentence.strip()
                        if not sentence:
                            continue
                
                        # Check if adding this sentence would exceed max_chunk_size
                        test_chunk = current_chunk + (" " if current_chunk else "") + sentence
                
                        if len(test_chunk) <= max_chunk_size:
                            current_chunk = test_chunk
                        else:
                            # Current chunk is full, save it and start new
                            if current_chunk:
                                chunks.append(current_chunk)
                    
                            # If single sentence is too long, split at commas/semicolons
                            if len(sentence) > max_chunk_size:
                                subsents = _split_long_sentence(sentence, max_chunk_size)
                                chunks.extend(subsents)
                                current_chunk = ""
                            else:
                                current_chunk = sentence
            
                    # Add remaining chunk
                    if current_chunk:
                        chunks.append(current_chunk)
        
                logger.info(f"Text chunked into {len(chunks)} chunks (max_size={max_chunk_size}, preserve_structure={preserve_structure})")
                return chunks
    
            except Exception as e:
                logger.error(f"Error chunking text: {e}")
                raise PreprocessingError(f"Failed to chunk text: {e}")
            - word_count: int (word count)
            - has_formula: bool (contains mathematical symbols)
            - academic_level: str (basic, intermediate, advanced)
    
    Examples:
        >>> meta = extract_metadata("# Introduction")
        >>> meta['type'] == 'heading'
        True
    """
    if not chunk or not isinstance(chunk, str):
        logger.warning("Empty or invalid chunk for metadata extraction")
        return {
            'type': 'unknown',
            'topic': context_topic or 'unknown',
            'char_count': 0,
            'word_count': 0,
            'has_formula': False,
            'academic_level': 'basic'
        }
    
    try:
        chunk = chunk.strip()
        char_count = len(chunk)
        word_count = len(chunk.split())
        
        # Detect content type
        content_type = _detect_content_type(chunk)
        
        # Extract topic
        topic = _extract_topic(chunk, context_topic)
        
        # Detect formulas
        has_formula = _has_formula(chunk)
        
        # Estimate academic level
        academic_level = _estimate_academic_level(chunk, has_formula, word_count)
        
        metadata = {
            'type': content_type,
            'topic': topic,
            'char_count': char_count,
            'word_count': word_count,
            'has_formula': has_formula,
            'academic_level': academic_level
        }
        
        logger.debug(f"Metadata extracted: type={content_type}, topic={topic}")
        return metadata
    
    except Exception as e:
        logger.error(f"Error extracting metadata: {e}")
        return {
            'type': 'unknown',
            'topic': context_topic or 'unknown',
            'char_count': len(chunk) if chunk else 0,
            'word_count': len(chunk.split()) if chunk else 0,
            'has_formula': False,
            'academic_level': 'basic'
        }


def _detect_content_type(chunk: str) -> str:
    """Detect content type (heading, formula, list, paragraph)."""
    chunk = chunk.strip()
    
    # Heading: short, title case, no ending punctuation
    if len(chunk) < 50 and not chunk.endswith(('.', '!', '?', ':', ';')):
        if chunk[0].isupper() and chunk.count(' ') < 10:
            if not re.search(r'[.!?;:]$', chunk):
                return 'heading'
    
    # Formula: contains math symbols
    if _has_formula(chunk):
        return 'formula'
    
    # List: starts with bullet or numbered
    if re.match(r'^[\s]*([-*•]|[\d]+\.)', chunk):
        return 'list'
    
    # Default: paragraph
    return 'paragraph'


def _extract_topic(chunk: str, context_topic: str = None) -> str:
    """Extract topic from chunk (first heading or sentence)."""
    chunk = chunk.strip()
    
    # Check for markdown heading
    if chunk.startswith('#'):
        topic = re.sub(r'^#+\s*', '', chunk).strip()
        return topic
    
    # Short title-case line as topic
    if len(chunk) < 100 and chunk[0].isupper() and '\n' not in chunk:
        if not chunk.endswith(('.', '!', '?')):
            return chunk
    
    # Extract first sentence as topic
    sentences = nltk.sent_tokenize(chunk)
    if sentences:
        first_sent = sentences[0].strip()
        if len(first_sent) < 100:
            return first_sent
    
    # Use context topic if available
    if context_topic:
        return context_topic
    
    # Fallback
    return chunk.split('\n')[0][:80] if chunk else 'unknown'


def _has_formula(text: str) -> bool:
    """Check if text contains mathematical formulas or equations."""
    formula_patterns = [
        r'[=+\-*/→⇒×÷]',  # Math operators
        r'[₀-₉₊₋₌]',       # Subscripts
        r'[⁰-⁹⁺⁻⁼]',       # Superscripts
        r'\b[A-Z]O\d',     # Chemical formulas (CO2, H2O, etc.)
        r'\$[^\$]*\$',      # LaTeX inline math
        r'\\left|\\right|\\frac|\\sqrt',  # LaTeX commands
    ]
    
    for pattern in formula_patterns:
        if re.search(pattern, text):
            return True
    
    return False
    
    Detects strong formula indicators while avoiding false positives from
    hyphenated prose (e.g., "well-known" should not be flagged as formula).
    
    Patterns checked:
    - Explicit operators: =, +, ×, ÷, →, ⇒, /
    - Subscripts/superscripts: ₀-₉, ⁰-⁹
    - Chemical formulas: CO2, H2O pattern
    - LaTeX delimiters: $ ... $, \command{...}
    - Math minus: - only when adjacent to digits/variables (e.g., "x-y", "-2")
    """
    formula_patterns = [
        r'=',                      # Equals sign
        r'[+×÷→⇒/]',              # Strong math operators (no plain -)
        r'[₀-₉₊₋₌]',              # Subscripts
        r'[⁰-⁹⁺⁻⁼]',              # Superscripts
        r'\b[A-Z]O\d',             # Chemical formulas (CO2, H2O, etc.)
        r'\$[^\$]*\$',             # LaTeX inline math
        r'\\left|\\right|\\frac|\\sqrt',  # LaTeX commands
        r'(?<=[a-zA-Z0-9])-(?=[a-zA-Z0-9])',  # Math minus: x-y, a-2 (not word-word)
    ]
    
    for pattern in formula_patterns:
        if re.search(pattern, text):
            return True
    
    return False


def _estimate_academic_level(chunk: str, has_formula: bool, word_count: int) -> str:
    """Estimate academic level (basic, intermediate, advanced)."""
    score = 0
    
    # Formula presence
    if has_formula:
        score += 2
    
    # Word count (longer = more advanced)
    if word_count > 50:
        score += 1
    if word_count > 100:
        score += 1
    
    # Technical terms
    technical_terms = [
        'hypothesis', 'methodology', 'analysis', 'synthesis', 'inference',
        'algorithm', 'optimization', 'differential', 'integral', 'theorem',
        'quantum', 'molecular', 'kinetic', 'entropy', 'catalyst'
    ]
    chunk_lower = chunk.lower()
    technical_count = sum(1 for term in technical_terms if term in chunk_lower)
    score += technical_count
    
    # Academic punctuation
    if ':' in chunk or ';' in chunk:
        score += 1
    
    # Determine level
    if score >= 4:
        return 'advanced'
    elif score >= 2:
        return 'intermediate'
    else:
        return 'basic'


def handle_markdown(text: str) -> str:
    """
    Handle markdown-formatted text by preserving structure and converting to readable format.
    
    This function:
    - Preserves markdown headings (#, ##, etc.) as structural elements
    - Keeps code blocks (triple backticks) as single units
    - Removes markdown syntax (bold *, italics _, links)
    - Normalizes markdown elements
    
    Args:
        text (str): Markdown text input
    
    Returns:
        str: Cleaned text with markdown structure preserved in plain text
    
    Examples:
        >>> markdown = "# Title\\n\\n**bold** text [link](url)"
        >>> clean = handle_markdown(markdown)
        >>> 'Title' in clean
        True
    """
    try:
        # Preserve code blocks
        code_blocks = []
        def preserve_code(match):
            code_blocks.append(match.group(0))
            return f"__CODE_BLOCK_{len(code_blocks)-1}__"
        
        text = re.sub(r'```[\s\S]*?```', preserve_code, text)
        
        # Remove markdown syntax but keep content
        text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)  # **bold** → bold
        text = re.sub(r'__([^_]+)__', r'\1', text)       # __bold__ → bold
        text = re.sub(r'\*([^\*]+)\*', r'\1', text)      # *italic* → italic
        text = re.sub(r'_([^_]+)_', r'\1', text)         # _italic_ → italic
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # [text](url) → text
        
        # Restore code blocks
        for idx, code in enumerate(code_blocks):
            text = text.replace(f"__CODE_BLOCK_{idx}__", code)
        
        return text
    
    except Exception as e:
        logger.error(f"Error handling markdown: {e}")
        return text


def handle_pdf_text(text: str) -> str:
    """
    Handle PDF-extracted text by removing common artifacts.
    
    This function:
    - Removes page numbers and headers/footers
    - Fixes hyphenation at line breaks (word-\nword → word)
    - Removes excessive blank lines
    - Normalizes spacing
    
    Args:
        text (str): PDF-extracted text (typically messy)
    
    Returns:
        str: Cleaned text with PDF artifacts removed
    
    Examples:
        >>> pdf_text = "word-\\nother text"
        >>> clean = handle_pdf_text(pdf_text)
        >>> 'wordother' in clean or 'word other' in clean
        True
    """
    try:
        # Fix hyphenation: word-\n → word
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        
        # Remove page numbers (common patterns: "Page 1", "1", centered numbers)
        text = re.sub(r'^[\s]*\d+[\s]*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'Page\s+\d+', '', text, flags=re.IGNORECASE)
        
        # Remove headers/footers (lines with repeated words or page references)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Skip likely header/footer: too short, all caps, or page reference
            if len(line) > 3 and not (len(line) < 20 and line.isupper()):
                cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        
        # Remove excessive blank lines
        text = re.sub(r'\n\n\n+', '\n\n', text)
        
        return text
    
    except Exception as e:
        logger.error(f"Error handling PDF text: {e}")
        return text


def preprocess_text(
    raw_text: str,
    max_chunk_size: int = 500,
    input_format: str = 'plain'
) -> List[Dict[str, Any]]:
    """
    Main preprocessing pipeline: clean, chunk, and extract metadata from text.
    
    This orchestrator function:
    1. Handles format-specific cleaning (markdown, PDF, plain text)
    2. Cleans and normalizes text
    3. Chunks text into semantic units
    4. Extracts metadata for each chunk (topic, type, formulas, etc.)
    5. Logs preprocessing statistics
    
    Args:
        raw_text (str): Raw input text (from file, markdown, or PDF extraction)
        max_chunk_size (int): Maximum characters per chunk. Default: 500
        input_format (str): Input format ('plain', 'markdown', 'pdf'). Default: 'plain'
    
    Returns:
        List[Dict[str, Any]]: List of chunks with metadata:
            Each item: {
                'text': str,
                'metadata': {
                    'type': str,
                    'topic': str,
                    'char_count': int,
                    'word_count': int,
                    'has_formula': bool,
                    'academic_level': str
                }
            }
    
    Examples:
        >>> text = "# Biology\\n\\nPhotosynthesis is..."
        >>> chunks = preprocess_text(text, input_format='markdown')
        >>> len(chunks) > 0
        True
        >>> 'metadata' in chunks[0]
        True
    """
    if not raw_text:
        logger.warning("Empty input text for preprocessing")
        return []
    
    if not isinstance(raw_text, str):
        raise PreprocessingError("Input must be a string")
    
    try:
        logger.info(f"Starting preprocessing: format={input_format}, size={len(raw_text)}")
        
        # Format-specific preprocessing
        if input_format == 'markdown':
            raw_text = handle_markdown(raw_text)
        elif input_format == 'pdf':
            raw_text = handle_pdf_text(raw_text)
        
        # Clean text
        cleaned = clean_text(raw_text)
        
        # Chunk text
        chunk_list = chunk_text(cleaned, max_chunk_size=max_chunk_size)
        
        # Extract metadata for each chunk
        result = []
        context_topic = None
        
        for chunk in chunk_list:
            metadata = extract_metadata(chunk, context_topic=context_topic)
            
            # Propagate topic context
            if metadata['type'] in ('heading', 'formula'):
                context_topic = metadata['topic']
            
            result.append({
                'text': chunk,
                'metadata': metadata
            })
        
        # Log statistics
        if result:
            avg_chunk_size = sum(len(c['text']) for c in result) / len(result)
            type_counts = {}
            for c in result:
                ctype = c['metadata']['type']
                type_counts[ctype] = type_counts.get(ctype, 0) + 1
            
            logger.info(
                f"Preprocessing complete: {len(result)} chunks, "
                f"avg_size={avg_chunk_size:.0f}. "
                f"Type distribution: {type_counts}"
            )
        
        return result
    
    except Exception as e:
        logger.error(f"Error in preprocessing pipeline: {e}")
        raise PreprocessingError(f"Preprocessing failed: {e}")


# Example usage
if __name__ == '__main__':
    # Configure logging for testing
    logging.basicConfig(level=logging.INFO)
    
    # Example: process sample academic text
    sample = """# Photosynthesis

Photosynthesis is the process by which green plants use sunlight to synthesize foods.

Key Formula:
6CO2 + 6H2O + light → C6H12O6 + 6O2

Light-dependent Reactions:
1. Occur in thylakoid membranes
2. Produce ATP and NADPH
3. Split water molecules

Important Concepts:
- Chloroplast: Organelle where photosynthesis occurs
- Chlorophyll: Green pigment that absorbs light
- Stomata: Pores for gas exchange
"""
    
    chunks = preprocess_text(sample, max_chunk_size=500)
    print(f"\nProcessed {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i} ({chunk['metadata']['type']}):")
        print(f"  Topic: {chunk['metadata']['topic']}")
        print(f"  Text: {chunk['text'][:80]}...")
        print()
