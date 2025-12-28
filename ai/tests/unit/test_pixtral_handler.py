import sys
import types
from pathlib import Path
from PIL import Image

import pytest


class _MockProcessor:
    def __init__(self):
        self.called_with = None

    def __call__(self, text, images, return_tensors):
        class _Inputs(dict):
            def to(self, device):
                return self
        self.called_with = {'text': text, 'images': images}
        return _Inputs()

    def batch_decode(self, outputs, skip_special_tokens=True):
        return ['Mocked OCR text']


class _MockModel:
    def __init__(self):
        self.device = None
        self.eval_called = False

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True

    def generate(self, **kwargs):
        return ['generated_ids']


@pytest.fixture(autouse=True)
def mock_transformers(monkeypatch):
    mock_proc = _MockProcessor()
    mock_model = _MockModel()

    class _AutoProcessor:
        @staticmethod
        def from_pretrained(model_id, cache_dir=None, trust_remote_code=True):
            return mock_proc

    class _AutoModel:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            return mock_model

    monkeypatch.setitem(sys.modules, 'transformers', types.SimpleNamespace(
        AutoProcessor=_AutoProcessor,
        AutoModelForCausalLM=_AutoModel,
    ))

    monkeypatch.setitem(sys.modules, 'torch', types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None),
        float16='float16',
        float32='float32',
    ))
    monkeypatch.setenv('PIXTRAL_MODEL', 'mistral-community/pixtral-12b')


def test_pixtral_ocr_extract_text(tmp_path):
    img_path = tmp_path / 'test.jpg'
    img = Image.new('RGB', (64, 64), color=(255, 255, 255))
    img.save(img_path)

    from modules.ocr.mistral_ocr import PixtralOCR

    handler = PixtralOCR.get_instance()
    res = handler.extract_text_from_path(str(img_path))

    assert isinstance(res, dict)
    assert res.get('text') == 'Mocked OCR text'
    assert res.get('device') is not None
    assert res.get('model')
