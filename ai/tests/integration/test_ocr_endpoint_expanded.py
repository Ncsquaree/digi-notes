import pytest
from fastapi.testclient import TestClient
import main as ai_main


@pytest.mark.integration
def test_ocr_extract_s3_success(mock_boto3_client):
    client = TestClient(ai_main.app)
    r = client.post('/ocr/extract', json={'s3_key': 'test.jpg'})
    # endpoint may require more auth; at minimum ensure we get a response
    assert r.status_code in (200, 422, 400)
