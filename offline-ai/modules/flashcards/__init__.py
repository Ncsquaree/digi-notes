"""Flashcard generation module (Phase 5).

Converts preprocessed text chunks with NER entities into study flashcards using:
1. Rule-based question templates (definition, why/how, difference, formula, application)
2. MobileBERT-SQuAD model for extractive answer generation (with rule-based fallback)
3. Difficulty mapping based on academic level, formula presence, entity count
4. Deduplication using Levenshtein similarity (threshold 0.8)
5. SQLite storage with embedding integration

Example:
    >>> from flashcards.generate import FlashcardGenerator
    >>> from utils.ner import EntityExtractor
    >>> from utils.embeddings import EmbeddingGenerator
    >>> 
    >>> ner = EntityExtractor(model_path="models/bert_model.tflite")
    >>> emb = EmbeddingGenerator(model_path="models/use-lite.tflite")
    >>> gen = FlashcardGenerator(db_path="offline_ai.db", ner_extractor=ner, embedding_generator=emb)
    >>> 
    >>> # Generate from preprocessed chunks
    >>> flashcards = gen.generate_from_chunks(chunks)
    >>> gen.store_flashcards(flashcards)
    >>> 
    >>> print(f"Generated {len(flashcards)} flashcards")

Classes:
    Flashcard: Dataclass representing a single flashcard
    FlashcardGenerator: Main class for generating and storing flashcards

Constants:
    QUESTION_TEMPLATES: Dict mapping question types to template patterns
"""

from dataclasses import dataclass
from typing import List, Optional
from .generate import FlashcardGenerator, Flashcard, QUESTION_TEMPLATES

__all__ = [
    'FlashcardGenerator',
    'Flashcard',
    'QUESTION_TEMPLATES',
]

__version__ = '1.0.0'
