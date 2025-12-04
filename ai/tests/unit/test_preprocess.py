import tempfile
from modules.ocr.preprocess import preprocess_image


def test_preprocess_basic(sample_image):
    # save to a temporary file and run preprocess with heavy ops disabled
    import os
    fp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    try:
        sample_image.save(fp.name, format='JPEG')
        res = preprocess_image(fp.name, enable_denoise=False, enable_threshold=False, enable_deskew=False)
        assert 'original' in res and 'processed' in res and 'metadata' in res
        assert res['metadata']['original_width'] == 100
    finally:
        try:
            os.unlink(fp.name)
        except Exception:
            pass
