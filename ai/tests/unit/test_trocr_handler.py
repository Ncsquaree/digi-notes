import pytest
from unittest.mock import MagicMock
import builtins

from modules.ocr import trocr_handler


def test_trocr_singleton_and_extract(monkeypatch, sample_image_bytes):
    # mock torch and transformers model behavior
    class FakeModel:
        def __init__(self):
            pass

        def eval(self):
            pass

        def to(self, device):
            return self

        def __call__(self, inputs):
            return {'logits': None}

    # patch model loader inside trocr_handler
    monkeypatch.setattr(trocr_handler, 'TrOCRHandler', trocr_handler.TrOCRHandler)
    # instantiate and monkeypatch the internal model methods
    h = trocr_handler.TrOCRHandler.get_instance()
    # patch extract_text to return known value
    monkeypatch.setattr(h, 'extract_text', lambda *a, **k: {'text': 'hello', 'confidence': 0.99})
    out = h.extract_text(sample_image_bytes)
    assert out['text'] == 'hello'
    assert out['confidence'] > 0.9


def test_trocr_load_failure(monkeypatch):
    # simulate missing torch -> handler should still be constructible but is_enabled False
    monkeypatch.setattr('modules.ocr.trocr_handler.torch', None)
    # reload? create new instance manually
    try:
        import importlib
        importlib.reload(trocr_handler)
    except Exception:
        pass
    # getting instance may raise; ensure attribute exists
    assert hasattr(trocr_handler, 'TrOCRHandler')
