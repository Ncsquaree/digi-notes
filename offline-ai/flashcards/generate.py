"""
Phase 5: Flashcard generation for the offline AI pipeline.

Features
--------
- Rule-based question templates (definition, why/how, difference, formula)
- MobileBERT-SQuAD extractive QA for answer extraction
- Quality filtering and deduplication
- SQLite storage with embedding links

Design notes
------------
- Reuses existing TFLite infrastructure (NER's MobileBERT, embeddings)
- Generates 2-5 Q/A pairs per chunk using academic templates
- Answers extracted via MobileBERT-SQuAD QA inference
- Falls back to heuristic extraction when model unavailable
- Deduplicates across chunks using normalized text comparison
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# Question templates for different entity/chunk types
QUESTION_TEMPLATES = {
    "definition": [
        "What is {entity}?",
        "Define {entity}.",
        "Explain the concept of {entity}."
    ],
    "why_how": [
        "Why does {topic} occur?",
        "How does {process} work?",
        "What causes {phenomenon}?"
    ],
    "difference": [
        "What is the difference between {entity1} and {entity2}?",
        "How do {entity1} and {entity2} differ?",
        "Compare {entity1} and {entity2}."
    ],
    "formula": [
        "What is the formula for {concept}?",
        "Write the equation for {process}.",
        "Express {relationship} mathematically."
    ],
    "application": [
        "What is an example of {concept}?",
        "How is {principle} applied?",
        "Where is {method} used?"
    ]
}

# Stop words to exclude from entity-based questions
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were'
}


@dataclass
class Flashcard:
    """Represents a generated flashcard."""
    question: str
    answer: str
    topic: str
    entities: List[str]
    embedding_id: Optional[int]
    difficulty: int
    source_type: str  # "rule_based" or "qa_extracted"
    confidence: float = 1.0


class FlashcardGenerator:
    """High-level helper to generate flashcards from chunks using rule-based + QA approach."""
    
    def __init__(
        self,
        *,
        db_path: str = ":memory:",
        ner_extractor: Optional[object] = None,
        embedding_generator: Optional[object] = None,
    ) -> None:
        """Initialize flashcard generator.
        
        Args:
            db_path: SQLite database path
            ner_extractor: EntityExtractor instance (for MobileBERT QA inference)
            embedding_generator: EmbeddingGenerator instance (for reference)
        """
        self.db_path = db_path
        self.ner_extractor = ner_extractor
        self.embedding_generator = embedding_generator
        self._db_conn: Optional[sqlite3.Connection] = None
    
    # ------------------------------------------------------------------
    # Question Generation
    # ------------------------------------------------------------------
    def _select_templates(self, chunk: dict) -> List[str]:
        """Select question templates based on chunk metadata.
        
        Args:
            chunk: Chunk dict with 'text', 'metadata'
        
        Returns:
            List of 2-3 question templates
        """
        metadata = chunk.get('metadata', {})
        entities = metadata.get('entities', [])
        
        selected_templates = []
        
        # Formula-focused chunks
        if metadata.get('has_formula', False):
            selected_templates.extend(QUESTION_TEMPLATES.get('formula', [])[:2])
        
        # Multi-entity chunks
        if len(entities) >= 2:
            selected_templates.extend(QUESTION_TEMPLATES.get('difference', [])[:1])
        
        # Advanced/complex chunks
        if metadata.get('academic_level') == 'advanced' and metadata.get('type') == 'paragraph':
            selected_templates.extend(QUESTION_TEMPLATES.get('why_how', [])[:1])
        
        # Application/example questions
        if metadata.get('type') in ['paragraph', 'heading']:
            selected_templates.extend(QUESTION_TEMPLATES.get('application', [])[:1])
        
        # Default: definition questions
        if not selected_templates:
            selected_templates.extend(QUESTION_TEMPLATES.get('definition', [])[:2])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_templates = []
        for template in selected_templates:
            if template not in seen:
                unique_templates.append(template)
                seen.add(template)
        
        return unique_templates[:3]  # Limit to 3 templates per chunk
    
    def _generate_rule_based_questions(self, chunk: dict) -> List[str]:
        """Generate question strings using templates.
        
        Args:
            chunk: Chunk dict with 'metadata' containing entities
        
        Returns:
            List of question strings
        """
        metadata = chunk.get('metadata', {})
        topic = metadata.get('topic', 'this topic')
        entities = metadata.get('entities', [])
        
        # Extract entity texts, filter stop words
        entity_texts = [
            e['text'] for e in entities 
            if e.get('text', '').lower() not in STOP_WORDS and len(e.get('text', '')) > 2
        ]
        
        templates = self._select_templates(chunk)
        questions = []
        
        for template in templates:
            try:
                # Fill template placeholders
                question = template
                
                # Replace {entity} with first entity
                if '{entity}' in question and entity_texts:
                    question = question.format(entity=entity_texts[0])
                
                # Replace {entity1}, {entity2} with two entities
                elif '{entity1}' in question and len(entity_texts) >= 2:
                    question = question.format(entity1=entity_texts[0], entity2=entity_texts[1])
                
                # Replace {topic}, {process}, {concept}, etc. with extracted topic or entity
                elif '{topic}' in question or '{process}' in question:
                    question = question.format(topic=topic, process=topic)
                elif '{concept}' in question:
                    question = question.format(concept=entity_texts[0] if entity_texts else topic)
                elif '{phenomenon}' in question:
                    question = question.format(phenomenon=topic)
                elif '{principle}' in question:
                    question = question.format(principle=entity_texts[0] if entity_texts else topic)
                elif '{method}' in question:
                    question = question.format(method=entity_texts[0] if entity_texts else topic)
                elif '{relationship}' in question:
                    question = question.format(relationship=topic)
                else:
                    # Skip template if we can't fill required placeholders
                    continue
                
                # Verify question is properly formatted (no dangling placeholders)
                if '{' not in question and '}' not in question:
                    questions.append(question)
            
            except (KeyError, IndexError):
                # Skip template if substitution fails
                continue
        
        return questions
    
    # ------------------------------------------------------------------
    # Answer Extraction
    # ------------------------------------------------------------------
    def _extract_answer_with_qa(
        self,
        question: str,
        chunk_text: str,
        confidence_threshold: float = 1.0
    ) -> Optional[Tuple[str, float]]:
        """Extract answer using MobileBERT-SQuAD or heuristic fallback.
        
        Args:
            question: Question string
            chunk_text: Source text for answer
            confidence_threshold: Minimum confidence (normalized logits)
        
        Returns:
            Tuple of (answer, confidence) or None if extraction fails
        """
        # Try MobileBERT inference if available
        if self.ner_extractor and self.ner_extractor.interpreter and self.ner_extractor.tokenizer:
            return self._extract_with_mobilebert(question, chunk_text, confidence_threshold)
        
        # Fallback to heuristic extraction
        return self._extract_heuristic(question, chunk_text)
    
    def _extract_with_mobilebert(
        self,
        question: str,
        chunk_text: str,
        confidence_threshold: float
    ) -> Optional[Tuple[str, float]]:
        """Extract answer using MobileBERT-SQuAD TFLite inference.
        
        Args:
            question: Question string
            chunk_text: Source text
            confidence_threshold: Minimum confidence threshold
        
        Returns:
            Tuple of (answer, confidence) or None
        """
        try:
            tokenizer = self.ner_extractor.tokenizer
            interpreter = self.ner_extractor.interpreter
            
            # Tokenize question and chunk
            question_tokens = tokenizer.encode(question, out_type=int)
            chunk_tokens = tokenizer.encode(chunk_text, out_type=int)
            
            # Combine: [CLS] question [SEP] chunk [SEP]
            max_len = 384
            input_ids = [101] + question_tokens[:128] + [102] + chunk_tokens[:max_len-len(question_tokens)-3] + [102]
            input_ids = input_ids + [0] * (max_len - len(input_ids))
            input_ids = input_ids[:max_len]
            
            # Run inference
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            input_array = np.array([input_ids], dtype=np.int32)
            interpreter.set_tensor(input_details[0]['index'], input_array)
            interpreter.invoke()
            
            # Get logits
            start_logits = interpreter.get_tensor(output_details[0]['index'])[0]
            end_logits = interpreter.get_tensor(output_details[1]['index'])[0]
            
            # Extract best answer span
            start_idx = int(np.argmax(start_logits))
            end_idx = int(np.argmax(end_logits))
            
            # Validate span
            if start_idx <= end_idx and 0 < end_idx - start_idx < 100:
                confidence = float((start_logits[start_idx] + end_logits[end_idx]) / 2)
                
                if confidence >= confidence_threshold:
                    # Decode span
                    span_tokens = input_ids[start_idx:end_idx+1]
                    answer_text = tokenizer.decode(span_tokens).strip()
                    
                    # Validate answer
                    if self._validate_answer(answer_text):
                        return (answer_text, min(confidence / 10.0, 1.0))  # Normalize confidence
        
        except Exception as e:
            logger.warning(f"MobileBERT extraction failed: {e}")
        
        return None
    
    def _extract_heuristic(self, question: str, chunk_text: str) -> Optional[Tuple[str, float]]:
        """Extract answer using heuristic rules (fallback).
        
        Args:
            question: Question string
            chunk_text: Source text
        
        Returns:
            Tuple of (answer, confidence) or None
        """
        sentences = chunk_text.split('.')
        
        # "What is X?" -> first sentence mentioning X
        if question.lower().startswith('what is'):
            entity = question.replace('What is', '').replace('Define', '').replace('?', '').strip()
            for sent in sentences:
                if entity.lower() in sent.lower():
                    answer = sent.strip()
                    if self._validate_answer(answer):
                        return (answer, 0.6)  # Medium confidence
        
        # "Why/How?" -> sentences with causal keywords
        if any(word in question.lower() for word in ['why', 'how', 'what causes']):
            causal_keywords = ['because', 'due to', 'causes', 'results in', 'leads to']
            for sent in sentences:
                if any(keyword in sent.lower() for keyword in causal_keywords):
                    answer = sent.strip()
                    if self._validate_answer(answer):
                        return (answer, 0.6)
        
        # "Formula?" -> extract formula-like patterns
        if 'formula' in question.lower() or 'equation' in question.lower():
            # Look for text with math symbols
            formula_pattern = r'[A-Z0-9]+\s*[=+×÷→⇒±]\s*[A-Za-z0-9+×÷→⇒±\s]+'
            match = re.search(formula_pattern, chunk_text)
            if match:
                answer = match.group(0).strip()
                if self._validate_answer(answer):
                    return (answer, 0.5)
        
        # Default: first sentence
        if sentences:
            answer = sentences[0].strip()
            if self._validate_answer(answer):
                return (answer, 0.5)
        
        return None
    
    def _validate_answer(self, answer: str) -> bool:
        """Validate answer quality.
        
        Args:
            answer: Answer string
        
        Returns:
            True if answer is valid
        """
        # Length check: 10-200 characters
        if len(answer) < 10 or len(answer) > 200:
            return False
        
        # Not just punctuation or numbers
        if not re.search(r'[a-zA-Z]', answer):
            return False
        
        # Not a single word (unless it's a proper noun/formula)
        words = answer.split()
        if len(words) == 1 and not any(c in answer for c in ['=', '+', '→']):
            return False
        
        return True
    
    # ------------------------------------------------------------------
    # Difficulty Mapping
    # ------------------------------------------------------------------
    def _map_difficulty(self, chunk_metadata: dict) -> int:
        """Map academic level to difficulty score (0-5).
        
        Args:
            chunk_metadata: Chunk metadata dict
        
        Returns:
            Difficulty (0-5)
        """
        academic_level = chunk_metadata.get('academic_level', 'intermediate')
        
        mapping = {
            'basic': 1,
            'intermediate': 2,
            'advanced': 4
        }
        
        base_difficulty = mapping.get(academic_level, 2)
        
        # Adjust based on entity count and formula presence
        entity_count = len(chunk_metadata.get('entities', []))
        has_formula = chunk_metadata.get('has_formula', False)
        
        difficulty = base_difficulty
        if entity_count >= 5:
            difficulty += 1
        if has_formula:
            difficulty += 1
        
        return min(difficulty, 5)  # Cap at 5
    
    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------
    def _normalize_question(self, question: str) -> str:
        """Normalize question for comparison.
        
        Args:
            question: Question string
        
        Returns:
            Normalized question
        """
        # Lowercase, remove punctuation, extra whitespace
        normalized = question.lower()
        normalized = re.sub(r'[?!.,;:]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between strings.
        
        Args:
            s1: First string
            s2: Second string
        
        Returns:
            Distance
        """
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
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
    
    def _similarity_ratio(self, s1: str, s2: str) -> float:
        """Calculate similarity ratio (0.0-1.0).
        
        Args:
            s1: First string
            s2: Second string
        
        Returns:
            Similarity (0.0-1.0)
        """
        distance = self._levenshtein_distance(s1, s2)
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1.0
        return 1.0 - (distance / max_len)
    
    def _deduplicate_flashcards(self, flashcards: List[Flashcard]) -> List[Flashcard]:
        """Remove duplicate flashcards.
        
        Args:
            flashcards: List of flashcards
        
        Returns:
            Deduplicated list
        """
        if not flashcards:
            return []
        
        unique = []
        for card in flashcards:
            # Check if similar question already exists
            is_duplicate = False
            for existing in unique:
                similarity = self._similarity_ratio(
                    self._normalize_question(card.question),
                    self._normalize_question(existing.question)
                )
                if similarity > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(card)
        
        return unique
    
    # ------------------------------------------------------------------
    # Main Generation Pipeline
    # ------------------------------------------------------------------
    def generate_from_chunk(self, chunk: dict) -> List[Flashcard]:
        """Generate flashcards for a single chunk.
        
        Args:
            chunk: Chunk dict with 'text', 'metadata'
        
        Returns:
            List of Flashcard objects
        """
        metadata = chunk.get('metadata', {})
        text = chunk.get('text', '')
        
        if not text or not metadata:
            return []
        
        flashcards = []
        topic = metadata.get('topic', 'unknown')
        entities = [e.get('text', '') for e in metadata.get('entities', [])]
        embedding_id = metadata.get('embedding_id')
        
        # Generate questions
        questions = self._generate_rule_based_questions(chunk)
        
        for question in questions:
            # Extract answer
            result = self._extract_answer_with_qa(question, text)
            
            if result:
                answer, confidence = result
                
                # Create flashcard
                flashcard = Flashcard(
                    question=question,
                    answer=answer,
                    topic=topic,
                    entities=entities,
                    embedding_id=embedding_id,
                    difficulty=self._map_difficulty(metadata),
                    source_type='qa_extracted' if confidence > 0.6 else 'rule_based',
                    confidence=confidence
                )
                flashcards.append(flashcard)
        
        return flashcards
    
    def generate_from_chunks(self, chunks: List[dict]) -> List[Flashcard]:
        """Generate flashcards from all chunks.
        
        Args:
            chunks: List of chunk dicts
        
        Returns:
            Deduplicated list of Flashcard objects
        """
        all_flashcards = []
        
        for chunk in chunks:
            flashcards = self.generate_from_chunk(chunk)
            all_flashcards.extend(flashcards)
        
        # Deduplicate across chunks
        unique_flashcards = self._deduplicate_flashcards(all_flashcards)
        
        return unique_flashcards
    
    # ------------------------------------------------------------------
    # SQLite Storage
    # ------------------------------------------------------------------
    def _ensure_db(self) -> sqlite3.Connection:
        """Ensure database connection."""
        if self._db_conn is None:
            self._db_conn = sqlite3.connect(self.db_path)
        return self._db_conn
    
    def store_flashcards(self, flashcards: List[Flashcard]) -> List[int]:
        """Store flashcards in SQLite database.
        
        Args:
            flashcards: List of Flashcard objects
        
        Returns:
            List of inserted flashcard IDs
        """
        conn = self._ensure_db()
        inserted_ids = []
        
        for flashcard in flashcards:
            cursor = conn.execute(
                """INSERT INTO flashcards (question, answer, difficulty, context, source_type, embedding_id)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    flashcard.question,
                    flashcard.answer,
                    flashcard.difficulty,
                    flashcard.topic,
                    flashcard.source_type,
                    flashcard.embedding_id
                )
            )
            inserted_ids.append(cursor.lastrowid)
        
        conn.commit()
        return inserted_ids
    
    def close(self) -> None:
        """Close database connection."""
        if self._db_conn:
            self._db_conn.close()
            self._db_conn = None


def generate_flashcards_from_chunks(
    chunks: List[dict],
    db_path: str = "offline_ai.db",
    ner_extractor: Optional[object] = None,
    embedding_generator: Optional[object] = None
) -> List[dict]:
    """Convenience function to generate and store flashcards.
    
    Args:
        chunks: Preprocessed chunks with metadata
        db_path: SQLite database path
        ner_extractor: EntityExtractor instance
        embedding_generator: EmbeddingGenerator instance
    
    Returns:
        List of flashcard dicts
    """
    gen = FlashcardGenerator(
        db_path=db_path,
        ner_extractor=ner_extractor,
        embedding_generator=embedding_generator
    )
    
    flashcards = gen.generate_from_chunks(chunks)
    gen.store_flashcards(flashcards)
    
    return [
        {
            'question': fc.question,
            'answer': fc.answer,
            'topic': fc.topic,
            'entities': fc.entities,
            'embedding_id': fc.embedding_id,
            'difficulty': fc.difficulty,
            'source_type': fc.source_type
        }
        for fc in flashcards
    ]
