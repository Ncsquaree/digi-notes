"""
Phase 4: Named Entity Recognition for the offline AI pipeline.

Features
--------
- MobileBERT-SQuAD TFLite inference for entity extraction via QA repurposing
- NLTK rule-based fallback when model unavailable
- Entity classification: Concept, Topic, Entity, Formula
- Deduplication with fuzzy matching
- Integration with preprocessing pipeline

Design notes
------------
- Repurposes MobileBERT-SQuAD (extractive QA) for entity extraction by framing
  entity detection as question answering
- Falls back to NLTK NER + noun phrase extraction when model unavailable
- Normalizes and deduplicates entities across chunks using Levenshtein distance
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import tflite_runtime.interpreter as tflite
except Exception:
    tflite = None

try:
    import sentencepiece as spm
except Exception:
    spm = None

try:
    import nltk
    from nltk import ne_chunk, pos_tag, word_tokenize
    from nltk.tree import Tree
except Exception:
    nltk = None

# Download required NLTK data on first run
if nltk:
    for resource in ['maxent_ne_chunker', 'words', 'averaged_perceptron_tagger', 'punkt']:
        try:
            nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'chunkers/{resource}' if 'chunker' in resource else f'taggers/{resource}')
        except LookupError:
            logger.info(f"Downloading NLTK {resource}...")
            nltk.download(resource, quiet=True)


class NERError(Exception):
    """Raised when NER fails irrecoverably."""


# Academic keywords for concept filtering
ACADEMIC_KEYWORDS = {
    'hypothesis', 'theorem', 'algorithm', 'principle', 'theory', 'method',
    'process', 'system', 'model', 'framework', 'approach', 'technique',
    'analysis', 'synthesis', 'equation', 'formula', 'reaction', 'mechanism',
    'structure', 'function', 'property', 'characteristic', 'behavior',
    'phenomenon', 'effect', 'law', 'rule', 'concept', 'definition'
}

# Stop words to exclude
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that',
    'these', 'those', 'it', 'its', 'they', 'them', 'their'
}


@dataclass
class Entity:
    """Represents an extracted entity."""
    text: str
    entity_type: str  # Concept, Topic, Entity, Formula
    confidence: float
    start_pos: int = -1
    end_pos: int = -1


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def similarity_ratio(s1: str, s2: str) -> float:
    """Calculate similarity ratio between two strings (0.0 to 1.0)."""
    distance = levenshtein_distance(s1.lower(), s2.lower())
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    return 1.0 - (distance / max_len)


def normalize_entity(entity: str) -> str:
    """Normalize entity string for comparison."""
    # Preserve formulas (don't lowercase if contains math symbols)
    if re.search(r'[=+×÷→⇒±∫∂Δ∑∏√²³⁴⁰¹⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉]', entity):
        return entity.strip()
    
    # Standard normalization for concepts
    normalized = entity.lower().strip()
    # Remove trailing punctuation but keep internal hyphens
    normalized = re.sub(r'[.,;:!?]+$', '', normalized)
    return normalized


def classify_entity_type(entity: str, context: str = "") -> str:
    """Classify entity into type: Concept, Topic, Entity, Formula.
    
    Args:
        entity: Entity text to classify
        context: Surrounding context for classification hints
    
    Returns:
        str: One of Concept, Topic, Entity, Formula
    """
    # Formula detection: contains math symbols
    if re.search(r'[=+×÷→⇒±∫∂Δ∑∏√]|[₀-₉₊₋₌]|[⁰-⁹⁺⁻⁼]|\d+[A-Z][A-Z0-9]*', entity):
        return "Formula"
    
    # Chemical formula pattern
    if re.match(r'^[A-Z][a-z]?[\d₀₁₂₃₄₅₆₇₈₉]*(?:[A-Z][a-z]?[\d₀₁₂₃₄₅₆₇₈₉]*)*$', entity):
        return "Formula"
    
    # Entity detection: capitalized proper noun (person, place, organization)
    if entity[0].isupper() and ' ' not in entity and len(entity) > 2:
        # Check if it's a proper noun in context
        if any(keyword in context.lower() for keyword in ['discovered', 'developed', 'proposed', 'named']):
            return "Entity"
    
    # Topic detection: short capitalized phrase (2-4 words)
    words = entity.split()
    if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if len(w) > 0):
        return "Topic"
    
    # Concept detection: academic keywords
    entity_lower = entity.lower()
    if any(keyword in entity_lower for keyword in ACADEMIC_KEYWORDS):
        return "Concept"
    
    # Default: Concept
    return "Concept"


def extract_noun_phrases(text: str) -> List[str]:
    """Extract noun phrases from text using POS tagging.
    
    Args:
        text: Input text
    
    Returns:
        List of noun phrases
    """
    if not nltk:
        return []
    
    try:
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        
        noun_phrases = []
        current_phrase = []
        
        # Extract noun phrase patterns (NN*, JJ+NN*)
        for i, (word, tag) in enumerate(pos_tags):
            if tag.startswith('NN') or tag.startswith('JJ'):
                current_phrase.append(word)
            else:
                if len(current_phrase) >= 1:
                    phrase = ' '.join(current_phrase)
                    if len(phrase) > 2 and phrase.lower() not in STOP_WORDS:
                        noun_phrases.append(phrase)
                current_phrase = []
        
        # Add final phrase
        if len(current_phrase) >= 1:
            phrase = ' '.join(current_phrase)
            if len(phrase) > 2 and phrase.lower() not in STOP_WORDS:
                noun_phrases.append(phrase)
        
        return noun_phrases
    
    except Exception as e:
        logger.warning(f"Noun phrase extraction failed: {e}")
        return []


def filter_academic_terms(entities: List[str]) -> List[str]:
    """Filter entities to keep only academic-relevant terms.
    
    Args:
        entities: List of entity strings
    
    Returns:
        Filtered list of academic terms
    """
    filtered = []
    
    for entity in entities:
        entity_lower = entity.lower()
        
        # Skip stop words
        if entity_lower in STOP_WORDS:
            continue
        
        # Skip very short entities (< 3 chars)
        if len(entity) < 3:
            continue
        
        # Skip purely numeric
        if entity.isdigit():
            continue
        
        # Keep if: formula, academic keyword, capitalized, or multi-word
        is_formula = re.search(r'[=+×÷→⇒±]', entity)
        has_keyword = any(kw in entity_lower for kw in ACADEMIC_KEYWORDS)
        is_capitalized = entity[0].isupper() if entity else False
        is_multiword = ' ' in entity
        
        if is_formula or has_keyword or is_capitalized or (is_multiword and len(entity) > 5):
            filtered.append(entity)
    
    return filtered


def nltk_fallback_ner(text: str) -> List[Entity]:
    """Rule-based NER using NLTK when TFLite model unavailable.
    
    Args:
        text: Input text
    
    Returns:
        List of Entity objects
    """
    if not nltk:
        logger.warning("NLTK not available for fallback NER")
        return []
    
    try:
        entities = []
        
        # Named entity recognition
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        ne_tree = ne_chunk(pos_tags)
        
        # Extract named entities
        for subtree in ne_tree:
            if isinstance(subtree, Tree):
                entity_text = ' '.join(word for word, tag in subtree.leaves())
                entity_type = "Entity"  # PERSON, ORG, GPE, etc.
                entities.append(Entity(
                    text=entity_text,
                    entity_type=entity_type,
                    confidence=0.7  # Medium confidence for NLTK
                ))
        
        # Extract noun phrases
        noun_phrases = extract_noun_phrases(text)
        for phrase in noun_phrases:
            # Avoid duplicates
            if not any(e.text.lower() == phrase.lower() for e in entities):
                entities.append(Entity(
                    text=phrase,
                    entity_type="Concept",
                    confidence=0.6
                ))
        
        # Filter by academic relevance
        entity_texts = [e.text for e in entities]
        filtered_texts = filter_academic_terms(entity_texts)
        entities = [e for e in entities if e.text in filtered_texts]
        
        return entities
    
    except Exception as e:
        logger.error(f"NLTK fallback NER failed: {e}")
        return []


class EntityExtractor:
    """High-level helper to load MobileBERT, extract entities, and classify them."""
    
    def __init__(
        self,
        *,
        model_path: Optional[Path] = None,
        spm_path: Optional[Path] = None,
        use_fallback: bool = False,
        auto_load: bool = True,
    ) -> None:
        # Default model paths
        if model_path is None and not use_fallback:
            default_model = Path(__file__).parent.parent / "models" / "mobilebert-squad.tflite"
            self.model_path = default_model if default_model.exists() else None
        else:
            self.model_path = Path(model_path) if model_path else None
        
        # Optional SentencePiece path
        if spm_path is None and not use_fallback:
            default_spm = Path(__file__).parent.parent / "models" / "mobilebert-squad.spm"
            self.spm_path = default_spm if default_spm.exists() else None
        else:
            self.spm_path = Path(spm_path) if spm_path else None
        
        self.use_fallback = use_fallback
        self.interpreter: Optional["tflite.Interpreter"] = None
        self.tokenizer: Optional["spm.SentencePieceProcessor"] = None
        
        if auto_load and not use_fallback:
            self.load()
    
    def load(self) -> None:
        """Load MobileBERT model and tokenizer."""
        if self.use_fallback:
            logger.info("Using NLTK fallback NER (use_fallback=True)")
            return
        
        self._load_tokenizer()
        self._load_model()
    
    def _load_tokenizer(self) -> None:
        """Load SentencePiece tokenizer for MobileBERT."""
        if self.spm_path is None:
            logger.warning("No SentencePiece tokenizer provided; will use fallback")
            return
        
        if spm is None:
            logger.warning("sentencepiece not installed; using fallback NER")
            return
        
        if not self.spm_path.exists():
            logger.warning(f"Tokenizer not found at {self.spm_path}; using fallback")
            return
        
        try:
            self.tokenizer = spm.SentencePieceProcessor(model_file=str(self.spm_path))
            logger.info(f"Loaded SentencePiece tokenizer from {self.spm_path}")
        except Exception as e:
            logger.warning(f"Failed to load tokenizer: {e}; using fallback")
    
    def _load_model(self) -> None:
        """Load MobileBERT-SQuAD TFLite model."""
        if self.model_path is None:
            msg = "No MobileBERT model found. Run 'python scripts/download_models.py'"
            logger.warning(msg)
            return
        
        if tflite is None:
            logger.warning("tflite_runtime not installed; using fallback NER")
            return
        
        if not self.model_path.exists():
            logger.warning(f"Model not found at {self.model_path}; using fallback")
            return
        
        try:
            interpreter = tflite.Interpreter(model_path=str(self.model_path))
            interpreter.allocate_tensors()
            self.interpreter = interpreter
            logger.info(f"Loaded MobileBERT model from {self.model_path}")
        except Exception as e:
            logger.warning(f"Failed to load model: {e}; using fallback")
    
    def extract_entities(self, text: str, context: str = "") -> List[Entity]:
        """Extract entities from text.
        
        Args:
            text: Input text to extract entities from
            context: Optional context for classification
        
        Returns:
            List of Entity objects
        """
        if not text or not text.strip():
            return []
        
        # Use MobileBERT if available
        if self.interpreter is not None and self.tokenizer is not None:
            return self._extract_with_mobilebert(text, context)
        
        # Fallback to NLTK
        return nltk_fallback_ner(text)
    
    def _extract_with_mobilebert(self, text: str, context: str) -> List[Entity]:
        """Extract entities using MobileBERT-SQuAD inference.
        
        Repurposes QA model for entity extraction by framing as:
        Question: "What are the key concepts?"
        Context: text
        Answer: extracted entities (spans)
        """
        try:
            # Implicit question for entity extraction
            question = "What are the key concepts, terms, and formulas?"
            
            # Tokenize question and context
            question_tokens = self.tokenizer.encode(question, out_type=int)
            context_tokens = self.tokenizer.encode(text, out_type=int)
            
            # Combine: [CLS] question [SEP] context [SEP]
            # MobileBERT max length: 384 tokens
            max_len = 384
            input_ids = [101] + question_tokens[:128] + [102] + context_tokens[:max_len-len(question_tokens)-3] + [102]
            input_ids = input_ids + [0] * (max_len - len(input_ids))  # Pad
            input_ids = input_ids[:max_len]
            
            # Prepare input
            input_details = self.interpreter.get_input_details()
            output_details = self.interpreter.get_output_details()
            
            # Set input tensor
            input_array = np.array([input_ids], dtype=np.int32)
            self.interpreter.set_tensor(input_details[0]['index'], input_array)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output logits (start and end positions)
            start_logits = self.interpreter.get_tensor(output_details[0]['index'])[0]
            end_logits = self.interpreter.get_tensor(output_details[1]['index'])[0]
            
            # Extract top answer spans
            entities = []
            top_k = 5  # Extract top 5 spans as entities
            
            for _ in range(top_k):
                start_idx = int(np.argmax(start_logits))
                end_idx = int(np.argmax(end_logits))
                
                # Validate span
                if start_idx <= end_idx and 0 < end_idx - start_idx < 50:
                    confidence = float(start_logits[start_idx] + end_logits[end_idx])
                    
                    # Skip low-confidence spans
                    if confidence < 1.0:
                        break
                    
                    # Decode span
                    span_tokens = input_ids[start_idx:end_idx+1]
                    entity_text = self.tokenizer.decode(span_tokens)
                    entity_text = entity_text.strip()
                    
                    # Filter noise
                    if len(entity_text) >= 2 and entity_text.lower() not in STOP_WORDS:
                        entity_type = classify_entity_type(entity_text, context)
                        entities.append(Entity(
                            text=entity_text,
                            entity_type=entity_type,
                            confidence=min(confidence / 10.0, 1.0),  # Normalize
                            start_pos=start_idx,
                            end_pos=end_idx
                        ))
                
                # Zero out to find next best span
                start_logits[start_idx] = -float('inf')
                end_logits[end_idx] = -float('inf')
            
            return entities
        
        except Exception as e:
            logger.warning(f"MobileBERT extraction failed: {e}; using fallback")
            return nltk_fallback_ner(text)
    
    def extract_from_chunks(self, chunks: List[dict]) -> List[dict]:
        """Extract entities from preprocessed chunks and add to metadata.
        
        Args:
            chunks: List of chunk dicts with 'text' and 'metadata'
        
        Returns:
            Updated chunks with 'entities' in metadata
        """
        for chunk in chunks:
            text = chunk.get('text', '')
            context = chunk.get('metadata', {}).get('topic', '')
            
            entities = self.extract_entities(text, context)
            
            # Add to metadata
            if 'metadata' not in chunk:
                chunk['metadata'] = {}
            
            chunk['metadata']['entities'] = [
                {
                    'text': e.text,
                    'type': e.entity_type,
                    'confidence': e.confidence
                }
                for e in entities
            ]
        
        return chunks
    
    def deduplicate_entities(self, entities: List[str]) -> List[str]:
        """Deduplicate entities using normalization and fuzzy matching.
        
        Args:
            entities: List of entity strings (possibly with duplicates)
        
        Returns:
            Deduplicated list of entities
        """
        if not entities:
            return []
        
        # Normalize and group
        normalized_map = {}  # normalized -> [original variants]
        for entity in entities:
            norm = normalize_entity(entity)
            if norm not in normalized_map:
                normalized_map[norm] = []
            normalized_map[norm].append(entity)
        
        # Select canonical form (most frequent or longest)
        unique_entities = []
        for norm, variants in normalized_map.items():
            # Count frequency
            freq = {}
            for v in variants:
                freq[v] = freq.get(v, 0) + 1
            
            # Select most frequent, or longest if tie
            canonical = max(variants, key=lambda x: (freq[x], len(x)))
            unique_entities.append(canonical)
        
        # Fuzzy matching for similar entities
        merged = []
        used = set()
        
        for i, e1 in enumerate(unique_entities):
            if i in used:
                continue
            
            # Find similar entities
            similar_group = [e1]
            for j, e2 in enumerate(unique_entities[i+1:], start=i+1):
                if j in used:
                    continue
                
                # Check similarity (threshold 0.8)
                if similarity_ratio(e1, e2) > 0.8:
                    similar_group.append(e2)
                    used.add(j)
            
            # Select canonical from group (longest)
            canonical = max(similar_group, key=len)
            merged.append(canonical)
            used.add(i)
        
        return merged


# Legacy function for backward compatibility
def extract_entities(text: str) -> list:
    """Extract key entities from text (legacy interface)."""
    extractor = EntityExtractor(use_fallback=True)
    entities = extractor.extract_entities(text)
    return [e.text for e in entities]
