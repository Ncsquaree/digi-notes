"""Flashcard generation using rule-based templates and MobileBERT-SQuAD (Phase 5).

This module generates Q&A flashcards from preprocessed text chunks using:
1. Rule-based question templates (definition, why/how, difference, formula, application)
2. MobileBERT-SQuAD TFLite for extractive answer generation
3. Difficulty mapping based on academic level and formula presence
4. Deduplication using Levenshtein similarity
5. SQLite storage with embedding integration

Architecture:
    - FlashcardGenerator: Main class coordinating template selection, answer extraction, 
      difficulty mapping, deduplication, and storage
    - Flashcard: Dataclass representing a single flashcard
    - QUESTION_TEMPLATES: Dict of template patterns by question type

Integration:
    - Receives chunks from preprocess with metadata (topic, type, academic_level, entities, embedding_id)
    - Uses ner_extractor for entity context in fallback answer extraction
    - Uses embedding_generator for optional semantic-based filtering
    - Stores in SQLite flashcards table with FK to embeddings table
"""

import sqlite3
import logging
import re
import difflib
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict

logger = logging.getLogger(__name__)

# Question templates by type - provides patterns for rule-based question generation
QUESTION_TEMPLATES = {
    'definition': [
        'What is {entity}?',
        'Define {entity}.',
        'Explain what {entity} means.',
        'What do you understand by {entity}?',
        'How would you define {entity}?',
    ],
    'why_how': [
        'Why is {entity} important?',
        'How does {entity} work?',
        'Explain how {entity} functions.',
        'Why do {entity_plural} use {entity}?',
        'What is the purpose of {entity}?',
        'How is {entity} implemented?',
    ],
    'difference': [
        'What is the difference between {entity1} and {entity2}?',
        'Compare {entity1} and {entity2}.',
        'How does {entity1} differ from {entity2}?',
        'What distinguishes {entity1} from {entity2}?',
        'Contrast {entity1} and {entity2}.',
    ],
    'formula': [
        'What is the formula for {entity}?',
        'State the equation for {entity}.',
        'What is the chemical formula for {entity}?',
        'How is {entity} calculated?',
        'What mathematical expression represents {entity}?',
    ],
    'application': [
        'How is {entity} applied in practice?',
        'What are the real-world applications of {entity}?',
        'Where is {entity} used?',
        'How can {entity} be applied to solve problems?',
        'Give an example of {entity} in use.',
    ],
}


@dataclass
class Flashcard:
    """Represents a single flashcard Q&A pair.
    
    Attributes:
        question: The question text (required)
        answer: The answer text (required)
        topic: The topic/chapter this flashcard belongs to
        entities: List of named entities mentioned in the Q&A
        embedding_id: FK to embeddings table for semantic search (optional)
        difficulty: Difficulty level 0-5 (0=trivial, 5=very hard)
        source_type: How the flashcard was generated (rule_based, qa_extracted, hybrid)
        created_at: Timestamp when flashcard was created
    """
    question: str
    answer: str
    topic: str = ""
    entities: List[str] = field(default_factory=list)
    embedding_id: Optional[int] = None
    difficulty: int = 2  # Default to intermediate
    source_type: str = "generated"
    created_at: str = ""
    
    def __post_init__(self):
        """Validate flashcard after initialization."""
        if not self.question or len(self.question) < 5:
            raise ValueError(f"Invalid question: '{self.question}'")
        if not self.answer or len(self.answer) < 10:
            raise ValueError(f"Invalid answer: '{self.answer}'")
        if not (0 <= self.difficulty <= 5):
            raise ValueError(f"Difficulty must be 0-5, got {self.difficulty}")


class FlashcardGenerator:
    """Generates flashcards from preprocessed text chunks.
    
    This class orchestrates the full pipeline:
    1. Select appropriate question templates based on chunk metadata
    2. Generate questions from templates
    3. Extract answers using MobileBERT-SQuAD or heuristics
    4. Map difficulty based on academic level and complexity
    5. Deduplicate similar flashcards
    6. Store in SQLite database
    
    Attributes:
        db_path: Path to SQLite database
        ner_extractor: Optional EntityExtractor instance (for enhanced fallback extraction)
        embedding_generator: Optional EmbeddingGenerator instance
        model_path: Path to MobileBERT-SQuAD TFLite model
        spm_path: Path to SentencePiece tokenizer model
        auto_load: Whether to auto-load models from default locations
        use_fallback: Use rule-based fallback when model unavailable
    """
    
    def __init__(self, 
                 db_path: str = "offline_ai.db",
                 ner_extractor: Optional[Any] = None,
                 embedding_generator: Optional[Any] = None,
                 model_path: Optional[str] = None,
                 spm_path: Optional[str] = None,
                 auto_load: bool = True,
                 use_fallback: bool = True):
        """Initialize FlashcardGenerator.
        
        Args:
            db_path: Path to SQLite database (created if not exists)
            ner_extractor: EntityExtractor instance for entity context (optional)
            embedding_generator: EmbeddingGenerator instance (optional)
            model_path: Path to MobileBERT-SQuAD TFLite model
            spm_path: Path to SentencePiece tokenizer model
            auto_load: Auto-load models from models/ directory
            use_fallback: Use rule-based extraction when model unavailable
            
        Raises:
            FileNotFoundError: If models not found and auto_load=False
        """
        self.db_path = db_path
        self.ner_extractor = ner_extractor
        self.embedding_generator = embedding_generator
        self.use_fallback = use_fallback
        self.model_path = model_path
        self.spm_path = spm_path
        self.auto_load = auto_load
        self.interpreter = None  # TFLite interpreter (lazy loaded)
        self.tokenizer = None  # SentencePiece tokenizer (lazy loaded)
        
        # Initialize database
        self._init_database()
        
        # Lazy-load model if auto_load=True
        if auto_load:
            self.load()
    
    def _init_database(self):
        """Initialize SQLite schema if not exists."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS flashcards (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    difficulty INTEGER DEFAULT 0,
                    context TEXT,
                    source_type TEXT DEFAULT 'generated',
                    embedding_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(embedding_id) REFERENCES embeddings(id)
                );
            """)
            conn.commit()
            conn.close()
            logger.info(f"Initialized flashcards database: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def load(self) -> bool:
        """Load MobileBERT-SQuAD model and tokenizer.
        
        Returns:
            True if model loaded successfully, False if fallback needed
            
        Raises:
            FileNotFoundError: If models not found and use_fallback=False
        """
        try:
            # Try to import and load TFLite model
            try:
                import tensorflow.lite as tflite
            except ImportError:
                import tflite_runtime.interpreter as tflite
            
            # Use provided paths or default to models/ directory
            model_path = self.model_path or "models/mobilebert-squad.tflite"
            spm_path = self.spm_path or "models/spm-10k-model.model"
            
            # Load TFLite interpreter
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            
            # Load SentencePiece tokenizer
            try:
                import sentencepiece as spm
                self.tokenizer = spm.SentencePieceProcessor()
                self.tokenizer.Load(spm_path)
            except ImportError:
                logger.warning("SentencePiece not available, using rule-based extraction")
                self.tokenizer = None
            
            logger.info(f"Loaded MobileBERT model from {model_path}")
            return True
            
        except (FileNotFoundError, ImportError) as e:
            if not self.use_fallback:
                raise FileNotFoundError(
                    f"MobileBERT model not found and fallback disabled: {e}"
                )
            logger.warning(f"Model loading failed, using rule-based fallback: {e}")
            return False
    
    def generate_from_chunk(self, chunk: Dict[str, Any]) -> List[Flashcard]:
        """Generate flashcards from a single preprocessed chunk.
        
        Args:
            chunk: Dict with 'text' and 'metadata' (topic, type, academic_level, entities, etc.)
            
        Returns:
            List of Flashcard objects generated from the chunk
        """
        if not chunk.get('text') or len(chunk['text'].strip()) < 50:
            return []  # Skip chunks with insufficient text
        
        flashcards = []
        metadata = chunk.get('metadata', {})
        topic = metadata.get('topic', 'Unknown')
        source_chunk_embedding_id = metadata.get('embedding_id')
        
        # Select appropriate question templates
        templates = self._select_templates(chunk)
        
        # Generate questions from templates
        questions = self._generate_rule_based_questions(chunk)
        
        # For each question, extract answer and create flashcard
        for question in questions:
            # Extract answer from chunk text
            answer_result = self._extract_answer_with_qa(question, chunk['text'])
            
            if answer_result is None:
                continue  # Skip if no answer found
            
            answer, confidence = answer_result
            
            # Compute embedding for flashcard (question + answer)
            flashcard_embedding_id = None
            if self.embedding_generator:
                try:
                    qa_text = f"{question} {answer}"
                    embedding_result = self.embedding_generator.embed_and_store_chunks(
                        [{'text': qa_text, 'metadata': {'type': 'flashcard', 'topic': topic}}]
                    )
                    if embedding_result:
                        flashcard_embedding_id = embedding_result[0].metadata.get('embedding_id')
                except Exception as e:
                    logger.debug(f"Failed to embed flashcard: {e}")
            
            # Create flashcard
            fc = Flashcard(
                question=question,
                answer=answer,
                topic=topic,
                entities=self._extract_entities_from_metadata(metadata),
                embedding_id=flashcard_embedding_id or source_chunk_embedding_id,
                difficulty=self._map_difficulty(metadata),
                source_type='qa_extracted' if confidence > 0.75 else 'rule_based'
            )
            
            flashcards.append(fc)
        
        return flashcards
    
    def generate_from_chunks(self, chunks: List[Dict[str, Any]]) -> List[Flashcard]:
        """Generate flashcards from multiple chunks.
        
        Args:
            chunks: List of preprocessed chunks
            
        Returns:
            Deduplicated list of Flashcard objects
        """
        all_flashcards = []
        
        for chunk in chunks:
            try:
                fcs = self.generate_from_chunk(chunk)
                all_flashcards.extend(fcs)
            except Exception as e:
                logger.error(f"Error generating flashcards from chunk: {e}")
                continue
        
        # Deduplicate
        unique_flashcards = self._deduplicate_flashcards(all_flashcards)
        
        logger.info(
            f"Generated {len(all_flashcards)} flashcards, "
            f"kept {len(unique_flashcards)} after deduplication"
        )
        
        return unique_flashcards
    
    def store_flashcards(self, flashcards: List[Flashcard]) -> List[int]:
        """Store flashcards in SQLite database.
        
        Args:
            flashcards: List of Flashcard objects to store
            
        Returns:
            List of inserted row IDs
            
        Raises:
            sqlite3.Error: If database operation fails
        """
        inserted_ids = []
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for fc in flashcards:
                cursor.execute("""
                    INSERT INTO flashcards 
                    (question, answer, difficulty, context, source_type, embedding_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    fc.question,
                    fc.answer,
                    fc.difficulty,
                    fc.topic,
                    fc.source_type,
                    fc.embedding_id
                ))
                inserted_ids.append(cursor.lastrowid)
            
            conn.commit()
            conn.close()
            
            logger.info(f"Stored {len(inserted_ids)} flashcards in database")
            return inserted_ids
            
        except sqlite3.Error as e:
            logger.error(f"Failed to store flashcards: {e}")
            raise
    
    def _select_templates(self, chunk: Dict[str, Any]) -> List[str]:
        """Select appropriate question templates based on chunk characteristics.
        
        Args:
            chunk: Chunk with metadata
            
        Returns:
            List of template patterns to use
        """
        templates = []
        metadata = chunk.get('metadata', {})
        
        has_formula = metadata.get('has_formula', False)
        has_entities = len(metadata.get('entities', [])) > 0
        entity_count = len(metadata.get('entities', []))
        
        # Always use definition templates
        templates.extend(QUESTION_TEMPLATES['definition'])
        
        # Use formula templates if chunk contains formula
        if has_formula:
            templates.extend(QUESTION_TEMPLATES['formula'])
        
        # Use why/how templates for intermediate+ content
        if metadata.get('academic_level') in ['intermediate', 'advanced']:
            templates.extend(QUESTION_TEMPLATES['why_how'])
        
        # Use difference templates if multiple entities
        if entity_count >= 2:
            templates.extend(QUESTION_TEMPLATES['difference'])
        
        # Use application templates for advanced content
        if metadata.get('academic_level') == 'advanced':
            templates.extend(QUESTION_TEMPLATES['application'])
        
        return templates if templates else QUESTION_TEMPLATES['definition']
    
    def _generate_rule_based_questions(self, chunk: Dict[str, Any]) -> List[str]:
        """Generate questions from chunk using template patterns.
        
        Args:
            chunk: Chunk with text and metadata
            
        Returns:
            List of generated questions (2-4 per chunk)
        """
        questions = []
        metadata = chunk.get('metadata', {})
        entities = metadata.get('entities', [])
        
        if not entities:
            # No entities, use topic as entity
            entity = metadata.get('topic', 'the concept')
        else:
            entity = entities[0]['text'] if isinstance(entities[0], dict) else entities[0]
        
        # Select templates for this chunk
        templates = self._select_templates(chunk)
        
        # Generate 2-3 questions using different templates
        import random
        selected_templates = random.sample(templates, min(3, len(templates)))
        
        for template in selected_templates:
            try:
                # Handle different template placeholders
                if '{entity}' in template:
                    question = template.format(entity=entity)
                elif '{entity1}' in template and len(entities) >= 2:
                    entity2 = entities[1]['text'] if isinstance(entities[1], dict) else entities[1]
                    question = template.format(entity1=entity, entity2=entity2)
                else:
                    continue  # Skip if template needs multiple entities
                
                # Validate question
                if self._validate_question(question):
                    questions.append(question)
            except Exception as e:
                logger.debug(f"Error generating question from template '{template}': {e}")
                continue
        
        return questions
    
    def _extract_answer_with_qa(self, question: str, text: str) -> Optional[Tuple[str, float]]:
        """Extract answer using MobileBERT-SQuAD or rule-based fallback.
        
        Args:
            question: The question to answer
            text: The context text to search for answer
            
        Returns:
            Tuple of (answer, confidence) or None if no answer found
        """
        # Try MobileBERT-SQuAD if model available
        if self.interpreter and self.tokenizer:
            try:
                answer, confidence = self._extract_with_mobilbert(question, text)
                if answer and self._validate_answer(answer):
                    return (answer, confidence)
            except Exception as e:
                logger.debug(f"MobileBERT extraction failed: {e}")
        
        # Fallback to rule-based extraction
        result = self._extract_heuristic(question, text)
        if result:
            return result
        
        return None
    
    def _extract_with_mobilbert(self, question: str, text: str) -> Tuple[Optional[str], float]:
        """Extract answer using MobileBERT-SQuAD TFLite model.
        
        Implementation:
        - Tokenize question and text separately
        - Allocate input based on question token count
        - Run through MobileBERT TFLite
        - Extract start/end positions from logits
        - Convert back to text span
        
        Args:
            question: Question text
            text: Context text
            
        Returns:
            Tuple of (answer_text, confidence) or (None, 0.0)
        """
        try:
            # Tokenize question and text
            q_tokens = self.tokenizer.EncodeAsIds(question)
            text_tokens = self.tokenizer.EncodeAsIds(text)
            
            # Compute token counts (use actual tokenized length for context allocation)
            q_tokens_count = len(q_tokens[:128])  # Cap question at 128 tokens
            q_tokens = q_tokens[:128]
            
            # Allocate remaining space for context (total max 384 tokens for BERT)
            max_tokens = 384
            remaining_for_context = max_tokens - q_tokens_count - 3  # -3 for [CLS], [SEP], [SEP]
            
            # Build input_ids from actual token counts
            context_tokens = text_tokens[:max(remaining_for_context, 50)]  # Keep at least 50 context tokens
            input_ids = [101] + q_tokens + [102] + context_tokens + [102]  # [CLS] + Q + [SEP] + C + [SEP]
            
            # Pad to max length
            while len(input_ids) < max_tokens:
                input_ids.append(0)
            input_ids = input_ids[:max_tokens]
            
            # Run inference
            input_data = {'input_ids': [[i for i in input_ids]]}
            self.interpreter.set_tensor(self.interpreter.get_input_details()[0]['index'], input_data['input_ids'])
            self.interpreter.invoke()
            
            # Get start/end logits
            start_logits = self.interpreter.get_tensor(self.interpreter.get_output_details()[0]['index'])
            end_logits = self.interpreter.get_tensor(self.interpreter.get_output_details()[1]['index'])
            
            # Find best answer span
            import numpy as np
            start_idx = int(np.argmax(start_logits[0]))
            end_idx = int(np.argmax(end_logits[0]))
            
            if start_idx >= end_idx or start_idx < q_tokens_count + 3:
                return (None, 0.0)  # Invalid span
            
            # Convert token indices back to text
            answer_tokens = input_ids[start_idx:end_idx + 1]
            answer = self.tokenizer.DecodeIds(answer_tokens)
            
            # Compute confidence from logits
            confidence = float(
                np.mean([start_logits[0][start_idx], end_logits[0][end_idx]])
            )
            confidence = min(1.0, max(0.0, confidence))  # Clamp to [0, 1]
            
            if self._validate_answer(answer):
                return (answer, confidence)
            else:
                return (None, 0.0)
            
        except Exception as e:
            logger.debug(f"MobileBERT extraction error: {e}")
            return (None, 0.0)
    
    def _extract_heuristic(self, question: str, text: str) -> Optional[Tuple[str, float]]:
        """Extract answer using rule-based heuristics.
        
        Strategies:
        - "What is X?" → Look for sentences starting with "X is"
        - "How does X?" → Look for sentences with "X" + verb patterns
        - "Why X?" → Look for "because", "due to", "caused by"
        
        Args:
            question: Question text
            text: Context text
            
        Returns:
            Tuple of (answer, confidence) or None
        """
        sentences = re.split(r'[.!?]+', text)
        
        # Extract key word from question
        words = question.lower().split()
        key_words = [w for w in words if len(w) > 3 and w not in {'what', 'does', 'how', 'why'}]
        
        best_answer = None
        best_score = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 20:
                continue
            
            # Score based on key word matches
            score = sum(1 for kw in key_words if kw in sentence.lower())
            
            if score > best_score:
                best_answer = sentence
                best_score = score
        
        if best_answer:
            # Validate and return
            if self._validate_answer(best_answer):
                confidence = min(0.7, best_score / len(key_words)) if key_words else 0.5
                return (best_answer, confidence)
        
        return None
    
    def _map_difficulty(self, metadata: Dict[str, Any]) -> int:
        """Map academic level and complexity to difficulty 0-5.
        
        Mapping:
        - basic + no formula → 1
        - basic + formula → 2
        - intermediate + no formula → 2
        - intermediate + formula → 3
        - advanced + no formula → 4
        - advanced + formula → 5
        
        Args:
            metadata: Chunk metadata with academic_level, has_formula, entities
            
        Returns:
            Difficulty level 0-5
        """
        level = metadata.get('academic_level', 'basic')
        has_formula = metadata.get('has_formula', False)
        entity_count = len(metadata.get('entities', []))
        
        # Base difficulty from academic level
        base = {
            'basic': 1,
            'intermediate': 2,
            'advanced': 4
        }.get(level, 2)
        
        # Adjust for formula
        if has_formula:
            base += 1
        
        # Adjust for multiple entities (indicates complexity)
        if entity_count > 2:
            base = min(5, base + 1)
        
        return min(5, max(1, base))
    
    def _deduplicate_flashcards(self, flashcards: List[Flashcard]) -> List[Flashcard]:
        """Remove duplicate or similar flashcards using Levenshtein similarity.
        
        Strategy:
        1. Normalize all questions (lowercase, no punctuation)
        2. For each question, find similar ones (similarity > 0.8)
        3. Keep only the one with best answer (longest/highest confidence)
        
        Args:
            flashcards: List of flashcards to deduplicate
            
        Returns:
            List of unique flashcards
        """
        if len(flashcards) <= 1:
            return flashcards
        
        # Group similar questions
        groups = []
        used = set()
        
        for i, fc in enumerate(flashcards):
            if i in used:
                continue
            
            group = [fc]
            used.add(i)
            
            norm_q = self._normalize_question(fc.question)
            
            for j, other_fc in enumerate(flashcards[i+1:], start=i+1):
                if j in used:
                    continue
                
                norm_other = self._normalize_question(other_fc.question)
                similarity = self._similarity_ratio(norm_q, norm_other)
                
                if similarity > 0.8:  # Similar questions
                    group.append(other_fc)
                    used.add(j)
            
            groups.append(group)
        
        # For each group, keep the best flashcard
        unique = []
        for group in groups:
            # Keep the one with longest answer (most informative)
            best = max(group, key=lambda x: len(x.answer))
            unique.append(best)
        
        return unique
    
    def _normalize_question(self, question: str) -> str:
        """Normalize question for similarity comparison.
        
        Args:
            question: Question text
            
        Returns:
            Normalized question (lowercase, no punctuation)
        """
        # Convert to lowercase
        normalized = question.lower()
        # Remove punctuation
        normalized = re.sub(r'[^\w\s]', '', normalized)
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        return normalized
    
    def _similarity_ratio(self, str1: str, str2: str) -> float:
        """Calculate Levenshtein-based similarity between two strings.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Similarity ratio 0.0-1.0
        """
        # Use difflib's SequenceMatcher
        matcher = difflib.SequenceMatcher(None, str1, str2)
        return matcher.ratio()
    
    def _validate_question(self, question: str) -> bool:
        """Validate that question meets minimum quality standards.
        
        Args:
            question: Question text
            
        Returns:
            True if valid, False otherwise
        """
        if not question or len(question) < 5:
            return False
        if len(question) > 200:
            return False
        if question.count('{') > 0 or question.count('}') > 0:
            return False  # Unfilled template
        if question.count('?') != 1:
            return False  # Should have exactly one question mark
        return True
    
    def _validate_answer(self, answer: str) -> bool:
        """Validate that answer meets minimum quality standards.
        
        Args:
            answer: Answer text
            
        Returns:
            True if valid, False otherwise
        """
        if not answer or len(answer) < 10:
            return False
        if len(answer) > 500:
            return False
        if answer.count('{') > 0 or answer.count('}') > 0:
            return False  # Unfilled template
        return True
    
    def _extract_entities_from_metadata(self, metadata: Dict[str, Any]) -> List[str]:
        """Extract entity strings from metadata for flashcard entities field.
        
        Args:
            metadata: Chunk metadata containing entities list
            
        Returns:
            List of entity text strings
        """
        entities = metadata.get('entities', [])
        result = []
        
        for entity in entities:
            if isinstance(entity, dict):
                result.append(entity.get('text', ''))
            elif isinstance(entity, str):
                result.append(entity)
        
        return [e for e in result if e]  # Remove empty strings
