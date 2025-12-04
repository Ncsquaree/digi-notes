"""
OCR module for handwritten text extraction.
Supports TrOCR (local model) and AWS Textract (cloud fallback).
"""
from .trocr_handler import TrOCRHandler, TrOCRModelError, TrOCRInferenceError
from .textract_handler import TextractHandler, TextractError
from .preprocess import preprocess_image, ImagePreprocessingError

__all__ = [
	'TrOCRHandler',
	'TextractHandler',
	'preprocess_image',
	'TrOCRModelError',
	'TrOCRInferenceError',
	'TextractError',
	'ImagePreprocessingError',
]
