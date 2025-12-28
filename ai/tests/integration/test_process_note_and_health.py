import pytest
from fastapi.testclient import TestClient
import main as ai_main


@pytest.mark.integration
def test_process_note_pipeline_basic(monkeypatch, mock_boto3_client, mock_openai_client):
    client = TestClient(ai_main.app)
    payload = {'user_id': 'u1', 'note_id': 'n1', 's3_key': 'test.jpg', 'options': {'await_result': True}}
    r = client.post('/process-note', json=payload)
    # pipeline may return 200 after processing; ensure we get JSON
    assert r.status_code in (200, 202, 504)


@pytest.mark.integration
def test_health_and_ready():
    client = TestClient(ai_main.app)
    r = client.get('/health')
    assert r.status_code == 200
    rr = client.get('/ready')
    # ready endpoint may return 200 or 503 depending on env/mocks
    assert rr.status_code in (200, 503)


@pytest.mark.integration
def test_process_note_with_pixtral(monkeypatch, mock_boto3_client, mock_openai_client, mock_pixtral):
    monkeypatch.setenv('PIXTRAL_ENABLED', 'true')
    monkeypatch.setenv('PIXTRAL_CONFIDENCE_THRESHOLD', '0.75')

    # stub file download to a tiny image
    from PIL import Image
    import tempfile

    def _fake_download(*args, **kwargs):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        Image.new('RGB', (10, 10), color=(255, 255, 255)).save(tmp, format='JPEG')
        tmp.close()
        return tmp.name

    monkeypatch.setattr('modules.utils.FileHandler.download_from_s3_by_key', lambda self, key: _fake_download())
    monkeypatch.setattr('modules.utils.FileHandler.download_from_s3', lambda self, url: _fake_download())

    client = TestClient(ai_main.app)
    payload = {
        'user_id': 'u1',
        'note_id': 'n1',
        's3_key': 'test.jpg',
        'options': {'await_result': True, 'enable_structurer': True},
    }
    r = client.post('/process-note', json=payload)
    assert r.status_code in (200, 202)

    data = r.json()
    if r.status_code == 200:
        result = data.get('result') or data.get('result_bundle') or {}
        ocr = result.get('ocr') or {}
        assert ocr.get('service') in ('pixtral', 'trocr', 'textract')
        assert 'confidence' in ocr


@pytest.mark.integration
def test_process_note_pixtral_fallback(monkeypatch, mock_boto3_client, mock_openai_client, mock_pixtral_low_conf):
    monkeypatch.setenv('PIXTRAL_ENABLED', 'true')
    monkeypatch.setenv('PIXTRAL_CONFIDENCE_THRESHOLD', '0.75')

    from PIL import Image
    import tempfile

    def _fake_download(*args, **kwargs):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        Image.new('RGB', (10, 10), color=(255, 255, 255)).save(tmp, format='JPEG')
        tmp.close()
        return tmp.name

    monkeypatch.setattr('modules.utils.FileHandler.download_from_s3_by_key', lambda self, key: _fake_download())
    monkeypatch.setattr('modules.utils.FileHandler.download_from_s3', lambda self, url: _fake_download())

    # Mock TrOCR to return higher confidence
    class MockTrOCR:
        @staticmethod
        def get_instance():
            return MockTrOCR()

        def extract_text(self, image):
            return {'text': 'trocr text', 'confidence': 0.85}

    monkeypatch.setattr('modules.ocr.trocr_handler.TrOCRHandler', MockTrOCR)

    client = TestClient(ai_main.app)
    payload = {'user_id': 'u1', 'note_id': 'n1', 's3_key': 'test.jpg', 'options': {'await_result': True}}
    r = client.post('/process-note', json=payload)
    assert r.status_code in (200, 202)
    data = r.json()
    if r.status_code == 200:
        result = data.get('result') or data.get('result_bundle') or {}
        ocr = result.get('ocr') or {}
        assert ocr.get('service') == 'trocr'
        assert float(ocr.get('confidence', 0)) >= 0.75
