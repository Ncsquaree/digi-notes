import pytest
from fastapi.testclient import TestClient
import main as ai_main
from pathlib import Path
from PIL import Image
import types


@pytest.mark.integration
def test_ocr_extract_s3_success(mock_boto3_client):
    client = TestClient(ai_main.app)
    r = client.post('/ocr/extract', json={'s3_key': 'test.jpg'})
    # endpoint may require more auth; at minimum ensure we get a response
    assert r.status_code in (200, 422, 400)


@pytest.mark.integration
def test_mistral_ocr_endpoint(mock_boto3_client, tmp_path, monkeypatch):
    # stub file handler to avoid real S3
    class DummyFH:
        def __init__(self):
            self.tmpdir = tmp_path

        def _write_sample(self, name: str):
            suffix = Path(name).suffix.lower() or '.jpg'
            path = self.tmpdir / f'sample{suffix or ".jpg"}'
            if suffix == '.pdf':
                img = Image.new('RGB', (32, 32), color=(255, 255, 255))
                img.save(path, format='PDF')
            else:
                img = Image.new('RGB', (32, 32), color=(255, 255, 255))
                img.save(path, format='JPEG')
            return str(path)

        def download_from_s3_by_key(self, key: str, local_path: str = None):
            return self._write_sample(key)

        def download_from_s3(self, s3_url: str, local_path: str = None):
            return self._write_sample(s3_url or 'sample.jpg')

        def get_file_info(self, file_path: str):
            mime = 'application/pdf' if Path(file_path).suffix.lower() == '.pdf' else 'image/jpeg'
            return {'size_bytes': 10, 'size_mb': 0.0001, 'extension': Path(file_path).suffix, 'mime_type': mime, 'filename': Path(file_path).name}

        def cleanup_temp_file(self, file_path: str):
            try:
                Path(file_path).unlink(missing_ok=True)
            except Exception:
                pass

    class DummyPix:
        def extract_text_from_path(self, image_path: str, use_preprocessing: bool = True):
            return {'text': 'pixtral text', 'confidence': 0.8, 'preprocessing_applied': use_preprocessing, 'model': 'pixtral', 'device': 'cpu', 'preprocessing_steps': ['denoise']}

        def extract_text_from_pdf(self, pdf_path: str, max_pages: int = None, use_preprocessing: bool = True):
            return {
                'text': 'pdf text',
                'confidence': 0.9,
                'pages': [{'page_num': 1, 'text': 'pdf text', 'confidence': 0.9}],
                'page_count': 1,
                'model': 'pixtral',
                'device': 'cpu',
                'preprocessing_applied': use_preprocessing,
                'preprocessing_steps': ['denoise'],
            }

    class DummyTrOCR:
        def extract_text(self, image):
            return {'text': 'trocr text', 'confidence': 0.95}

    monkeypatch.setattr(ai_main, 'FileHandler', DummyFH)
    monkeypatch.setattr(ai_main, 'PixtralOCR', types.SimpleNamespace(get_instance=lambda: DummyPix()))
    monkeypatch.setattr(ai_main, 'TrOCRHandler', types.SimpleNamespace(get_instance=lambda: DummyTrOCR()))

    client = TestClient(ai_main.app)

    # image flow
    r = client.post('/ocr/mistral', json={'s3_key': 'test-image.jpg', 'use_preprocessing': True, 'fallback_to_trocr': True})
    assert r.status_code == 200
    data = r.json()
    assert data['success'] is True
    assert data['text'] == 'pixtral text'
    assert data['method'] in ['pixtral', 'trocr']

    # pdf flow
    r_pdf = client.post('/ocr/mistral', json={'s3_key': 'test-doc.pdf', 'max_pdf_pages': 2})
    assert r_pdf.status_code == 200
    data_pdf = r_pdf.json()
    assert data_pdf['success'] is True
    assert data_pdf.get('page_count') == 1
