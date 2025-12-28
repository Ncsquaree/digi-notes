import os
import json
import uuid
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from dotenv import load_dotenv

# load test env first
env_path = Path(__file__).resolve().parents[1] / '.env.test'
if env_path.exists():
    load_dotenv(env_path)

os.environ.setdefault('TESTING', '1')


@pytest.fixture(autouse=True)
def silence_logger(monkeypatch):
    # Patch any project logger acquisition to avoid noisy logs
    try:
        import modules.utils.logger as logger_mod
        monkeypatch.setattr(logger_mod, 'get_logger', lambda *a, **k: MagicMock())
    except Exception:
        pass
    yield


@pytest.fixture
def sample_image():
    from PIL import Image
    img = Image.new('RGB', (100, 100), color=(255, 255, 255))
    return img


@pytest.fixture
def sample_image_bytes(sample_image):
    from io import BytesIO
    buf = BytesIO()
    sample_image.save(buf, format='JPEG')
    buf.seek(0)
    return buf.getvalue()


@pytest.fixture
def sample_parsed_content():
    return {
        'topics': [{'id': 't1', 'title': 'Photosynthesis', 'subtopics': []}],
        'formulas': [{'id': 'f1', 'latex': 'E=mc^2', 'description': 'energy-mass'}],
        'concepts': [{'id': 'c1', 'name': 'Chlorophyll', 'definition': 'pigment'}],
        'questions': [{'id': 'q1', 'text': 'What is photosynthesis?', 'difficulty': 2}]
    }


@pytest.fixture
def mock_openai_client(monkeypatch):
    # delegate to detailed mock helper to produce realistic responses, including function_call
    try:
        from tests.fixtures.mock_openai import mock_openai_create
    except Exception:
        try:
            from ai.tests.fixtures.mock_openai import mock_openai_create
        except Exception:
            mock_openai_create = None
    if mock_openai_create:
        try:
            import openai
            monkeypatch.setattr(openai.ChatCompletion, 'create', staticmethod(mock_openai_create))
        except Exception:
            pass
    return True


@pytest.fixture
def mock_pixtral(monkeypatch):
    class MockPixtral:
        @staticmethod
        def get_instance():
            return MockPixtral()

        def extract_text(self, image):
            return {
                'text': 'Sample OCR text from Pixtral',
                'confidence': 0.92,
                'blocks': [],
                'model': 'pixtral-12b',
                'device': 'cpu',
                'preprocessing_steps': ['denoise', 'threshold'],
            }

    monkeypatch.setattr('modules.ocr.mistral_ocr.PixtralOCR', MockPixtral)
    return MockPixtral


@pytest.fixture
def mock_pixtral_low_conf(monkeypatch):
    class MockPixtralLow:
        @staticmethod
        def get_instance():
            return MockPixtralLow()

        def extract_text(self, image):
            return {
                'text': 'Low quality OCR',
                'confidence': 0.5,
                'blocks': [],
                'model': 'pixtral-12b',
                'device': 'cpu',
            }

    monkeypatch.setattr('modules.ocr.mistral_ocr.PixtralOCR', MockPixtralLow)
    return MockPixtralLow


@pytest.fixture
def mock_boto3_client(monkeypatch):
    from unittest.mock import MagicMock

    class MockS3:
        def generate_presigned_url(self, ClientMethod, Params=None, ExpiresIn=3600):
            return f'https://s3.test/{Params.get("Key")}'

        def get_object(self, Bucket, Key):
            return {'Body': b'fake', 'ContentLength': 4, 'ContentType': 'image/jpeg'}

    def fake_client(name, *a, **kw):
        # prefer dedicated fixture module if present
        try:
            from tests.fixtures.mock_aws import fake_boto3_client as fb
        except Exception:
            try:
                from ai.tests.fixtures.mock_aws import fake_boto3_client as fb
            except Exception:
                fb = None
        if fb:
            return fb(name, *a, **kw)
        if name == 's3':
            return MockS3()
        return MagicMock()

    monkeypatch.setattr('boto3.client', fake_client)
    return True


@pytest.fixture
def mock_redis_client(monkeypatch):
    class MockRedis:
        def __init__(self):
            self.store = {}

        def get(self, k):
            return self.store.get(k)

        def set(self, k, v, ex=None):
            self.store[k] = v

        def delete(self, k):
            self.store.pop(k, None)

        def ping(self):
            return True

    monkeypatch.setattr('redis.Redis', lambda *a, **k: MockRedis())
    return True


@pytest.fixture
def mock_postgres_pool(monkeypatch):
    class MockCursor:
        def execute(self, q, p=None):
            self._last = (q, p)

        def fetchone(self):
            return None

        def fetchall(self):
            return []

        def close(self):
            pass

    class MockConn:
        def cursor(self):
            return MockCursor()

        def commit(self):
            pass

        def rollback(self):
            pass

        def close(self):
            pass

    class MockPool:
        def getconn(self):
            return MockConn()

        def putconn(self, conn):
            pass

        def closeall(self):
            pass

    monkeypatch.setattr('psycopg2.pool.SimpleConnectionPool', lambda *a, **k: MockPool())
    return True
