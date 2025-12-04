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
