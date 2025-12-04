"""
Flashcard generation module with spaced repetition (SM-2 algorithm).
Generates Q&A pairs from academic content using LLM.
"""

from .spaced_repetition import (
	FlashcardGenerator,
	FlashcardItem,
	GenerateFlashcardsRequest,
	GenerateFlashcardsResponse,
	generate_flashcards,
	FlashcardGeneratorError,
	FlashcardAPIError,
	FlashcardValidationError,
	FlashcardTimeoutError,
)

__all__ = [
	'FlashcardGenerator',
	'FlashcardItem',
	'GenerateFlashcardsRequest',
	'GenerateFlashcardsResponse',
	'generate_flashcards',
	'FlashcardGeneratorError',
	'FlashcardAPIError',
	'FlashcardValidationError',
	'FlashcardTimeoutError',
]
