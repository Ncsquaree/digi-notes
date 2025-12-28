import json
import sys
import types
from pathlib import Path
from PIL import Image

import pytest


class _MockProcessor:
    def __init__(self):
        self.called_with = None
        self.call_idx = 0

    def __call__(self, text, images, return_tensors):
        class _Inputs(dict):
            def to(self, device):
                return self
        self.called_with = {'text': text, 'images': images}
        return _Inputs()

    def batch_decode(self, outputs, skip_special_tokens=True):
        self.call_idx += 1
        payload = [{
            'text': f'Mocked OCR text {self.call_idx}',
            'bbox': {'x': 0.1, 'y': 0.1, 'width': 0.2, 'height': 0.1}
        }]
        return [json.dumps(payload)]


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
def mock_dependencies(monkeypatch):
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

    # mock pdf2image
    def _fake_convert(path):
        img1 = Image.new('RGB', (32, 32), color=(255, 255, 255))
        img2 = Image.new('RGB', (32, 32), color=(200, 200, 200))
        return [img1, img2]

    monkeypatch.setitem(sys.modules, 'pdf2image', types.SimpleNamespace(convert_from_path=_fake_convert))
    monkeypatch.setenv('PIXTRAL_MODEL', 'mistral-community/pixtral-12b')

    # mock preprocessing
    def _fake_preprocess(path):
        img = Image.open(path).convert('RGB')
        return {'processed': img, 'steps_applied': ['denoise', 'deskew']}

    import modules.ocr.mistral_ocr as pix_mod
    monkeypatch.setattr(pix_mod, 'convert_from_path', _fake_convert)
    monkeypatch.setattr(pix_mod, 'preprocess_image', _fake_preprocess)
    monkeypatch.setenv('PIXTRAL_PREPROCESSING_ENABLED', 'true')
    return {'proc': mock_proc, 'model': mock_model}


def test_pixtral_ocr_extract_text(tmp_path):
    img_path = tmp_path / 'test.jpg'
    img = Image.new('RGB', (64, 64), color=(255, 255, 255))
    img.save(img_path)

    from modules.ocr.mistral_ocr import PixtralOCR

    handler = PixtralOCR.get_instance()
    res = handler.extract_text_from_path(str(img_path))
    handler.cleanup()

    assert isinstance(res, dict)
    assert 'Mocked OCR text' in res.get('text')
    assert res.get('blocks')
    assert res.get('preprocessing_applied') is True


def test_pixtral_extract_from_pdf(tmp_path):
    pdf_path = tmp_path / 'doc.pdf'
    img = Image.new('RGB', (64, 64), color=(255, 255, 255))
    img.save(pdf_path, format='PDF')

    from modules.ocr.mistral_ocr import PixtralOCR

    handler = PixtralOCR.get_instance()
    res = handler.extract_text_from_pdf(str(pdf_path), max_pages=2, use_preprocessing=True)
    handler.cleanup()

    assert res.get('page_count') == 2
    assert len(res.get('pages')) == 2
    assert 'Mocked OCR text' in res.get('text')


def test_pixtral_confidence_calculation(tmp_path):
    from modules.ocr.mistral_ocr import PixtralOCR

    handler = PixtralOCR.get_instance()
    low = handler._calculate_confidence('abc', Image.new('RGB', (10, 10), color=(0, 0, 0)))
    high = handler._calculate_confidence('A' * 100, Image.new('RGB', (10, 10), color=(0, 0, 0)))
    handler.cleanup()

    assert low < high
    assert 0.3 <= low <= 1.0


def test_pixtral_pdf_page_limit(tmp_path, monkeypatch):
    pdf_path = tmp_path / 'doc_limit.pdf'
    img = Image.new('RGB', (64, 64), color=(255, 255, 255))
    img.save(pdf_path, format='PDF')

    def _fake_convert(path):
        return [Image.new('RGB', (32, 32), color=(255, 255, 255)) for _ in range(3)]

    import modules.ocr.mistral_ocr as pix_mod
    monkeypatch.setattr(pix_mod, 'convert_from_path', _fake_convert)
    monkeypatch.setenv('PIXTRAL_MAX_PDF_PAGES', '1')

    from modules.ocr.mistral_ocr import PixtralOCR

    handler = PixtralOCR.get_instance()
    res = handler.extract_text_from_pdf(str(pdf_path))
    handler.cleanup()

    assert res.get('page_count') == 1
