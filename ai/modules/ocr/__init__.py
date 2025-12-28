"""
OCR module for handwritten text extraction.
Supports TrOCR (local model), Mistral Pixtral (local vision-language), and AWS Textract (cloud fallback).
"""
from .trocr_handler import TrOCRHandler, TrOCRModelError, TrOCRInferenceError
from .mistral_ocr import PixtralOCR, PixtralModelError, PixtralInferenceError
from .textract_handler import TextractHandler, TextractError
from .preprocess import preprocess_image, ImagePreprocessingError
from .ocr_structurer import (
	StructuredDocument,
	Section,
	Paragraph,
	ListItem,
	Metadata as OCRMetadata,
	structure_document,
	flatten_structured_text,
)

__all__ = [
	'TrOCRHandler',
	'PixtralOCR',
	'TextractHandler',
	'preprocess_image',
	'TrOCRModelError',
	'TrOCRInferenceError',
	'PixtralModelError',
	'PixtralInferenceError',
	'TextractError',
	'ImagePreprocessingError',
	'StructuredDocument',
	'Section',
	'Paragraph',
	'ListItem',
	'OCRMetadata',
	'structure_document',
	'flatten_structured_text',
]
