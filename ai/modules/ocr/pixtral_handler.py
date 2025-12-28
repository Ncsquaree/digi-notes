"""Thin alias module for backward compatibility.

Use modules.ocr.mistral_ocr.PixtralOCR for local Pixtral inference.
"""
from .mistral_ocr import PixtralOCR, PixtralModelError, PixtralInferenceError

# Backwards-compat entry point
PixtralHandler = PixtralOCR
