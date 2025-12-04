import pytest
from fastapi.testclient import TestClient
import main as ai_main


@pytest.mark.integration
def test_parse_semantic_success(mock_openai_client):
    client = TestClient(ai_main.app)
    r = client.post('/parse/semantic', json={'text': 'Photosynthesis occurs in plants.'})
    assert r.status_code == 200
    data = r.json()
    assert data.get('success') is True
    assert 'parsed_content' in data


@pytest.mark.integration
def test_parse_semantic_empty_text():
    client = TestClient(ai_main.app)
    r = client.post('/parse/semantic', json={'text': '   '})
    assert r.status_code == 400


@pytest.mark.integration
def test_summarize_endpoint(mock_openai_client, sample_parsed_content):
    client = TestClient(ai_main.app)
    body = {'parsed_content': sample_parsed_content(), 'mode': 'brief'}
    r = client.post('/summarize', json=body)
    assert r.status_code == 200
    j = r.json()
    assert j.get('success') is True

    # invalid mode
    r2 = client.post('/summarize', json={'parsed_content': sample_parsed_content(), 'mode': 'bad'})
    assert r2.status_code == 400
