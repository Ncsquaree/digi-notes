import pytest
from fastapi.testclient import TestClient
import main as ai_main


@pytest.mark.integration
def test_flashcards_and_tools_endpoints(mock_openai_client, sample_parsed_content):
    client = TestClient(ai_main.app)
    body = {'parsed_content': sample_parsed_content(), 'count': 2}
    r = client.post('/flashcards/generate', json=body)
    assert r.status_code == 200
    j = r.json()
    assert j.get('success') is True

    # quiz
    qb = {'parsed_content': sample_parsed_content(), 'question_count': 2}
    rq = client.post('/tools/generate-quiz', json=qb)
    assert rq.status_code == 200

    # mindmap
    mb = {'parsed_content': sample_parsed_content(), 'options': {}}
    rm = client.post('/tools/generate-mindmap', json=mb)
    assert rm.status_code == 200
