from fastapi.testclient import TestClient
from modules import utils
import main as ai_main


def test_health_endpoint():
    client = TestClient(ai_main.app)
    r = client.get('/health')
    assert r.status_code == 200
    data = r.json()
    assert data.get('status') == 'ok'
