import pytest
from unittest.mock import patch

from modules.utils import FileHandler, FileDownloadError


def test_s3_download(monkeypatch, mock_boto3_client, tmp_path):
    # simulate download by mocking boto3 s3 get_object
    fh = FileHandler()
    # use generate_presigned_url path
    url = fh.get_presigned_url('bucket', 'key')
    assert 'https://' in url
