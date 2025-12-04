from modules.ocr.textract_handler import TextractHandler, TextractError


def test_textract_extract_local(monkeypatch, tmp_path, mock_boto3_client):
    # write a small image file
    p = tmp_path / 'img.jpg'
    from PIL import Image
    Image.new('RGB', (50, 50), color=(255, 255, 255)).save(p)

    # ensure boto3.client is patched by mock_boto3_client fixture
    handler = TextractHandler()
    assert hasattr(handler, 'extract_text_from_local')
    # call and expect a dict with text
    out = handler.extract_text_from_local(str(p))
    assert isinstance(out, dict)
    assert 'text' in out
