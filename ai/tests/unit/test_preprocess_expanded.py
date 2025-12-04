import numpy as np
import cv2
import tempfile
import os
import pytest

from modules.ocr.preprocess import denoise_image, threshold_image, deskew_image, preprocess_image, ImagePreprocessingError


def test_denoise_and_threshold(sample_image):
    # create noisy image
    import numpy as np
    arr = np.array(sample_image.convert('RGB'))
    noise = (np.random.randn(*arr.shape) * 10).astype('uint8')
    noisy = (arr + noise) % 256
    # denoise expects BGR
    bgr = cv2.cvtColor(noisy, cv2.COLOR_RGB2BGR)
    den = denoise_image(bgr)
    assert den is not None
    # threshold adaptive and otsu
    thresh_a = threshold_image(den, method='adaptive')
    assert thresh_a.dtype == den.dtype or thresh_a.dtype == den.dtype
    thresh_o = threshold_image(den, method='otsu')
    assert thresh_o is not None


def test_deskew_identity(sample_image):
    arr = cv2.cvtColor(np.array(sample_image), cv2.COLOR_RGB2BGR)
    rotated, angle = deskew_image(arr)
    assert isinstance(angle, float)
    assert rotated is not None


def test_preprocess_image_path_and_options(sample_image, monkeypatch):
    # save to temp file
    fp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    try:
        sample_image.save(fp.name, format='JPEG')
        res = preprocess_image(fp.name, enable_denoise=True, enable_threshold=True, enable_deskew=False)
        assert 'original' in res and 'processed' in res and 'metadata' in res
        # test with deskew enabled
        res2 = preprocess_image(fp.name, enable_denoise=False, enable_threshold=False, enable_deskew=True)
        assert 'steps_applied' in res2
    finally:
        os.unlink(fp.name)


def test_preprocess_invalid_path():
    with pytest.raises(ImagePreprocessingError):
        preprocess_image('nonexistent.jpg')
