"""Image preprocessing utilities for OCR.

Functions:
- denoise_image: remove noise using OpenCV fastNlMeansDenoising
- threshold_image: adaptive or Otsu binarization
- deskew_image: detect and correct skew angle
- preprocess_image: orchestrates steps and returns original/processed PIL images and metadata

Exceptions:
- ImagePreprocessingError

Environment variables:
- PREPROCESSING_ENABLED (default true)
- PREPROCESSING_DENOISE (default true)
- PREPROCESSING_THRESHOLD (default true)
- PREPROCESSING_DESKEW (default true)
- MIN_IMAGE_DIMENSION (default 50)
- MAX_IMAGE_DIMENSION (default 10000)
"""
from __future__ import annotations

import os
import time
import math
from typing import Tuple, Dict, Any, List

import cv2
import numpy as np
from PIL import Image

from modules.utils import get_logger

LOG = get_logger()

# Environment-driven defaults
PREPROCESSING_ENABLED = os.getenv('PREPROCESSING_ENABLED', 'true').lower() in ('1', 'true', 'yes')
PREPROCESSING_DENOISE = os.getenv('PREPROCESSING_DENOISE', 'true').lower() in ('1', 'true', 'yes')
PREPROCESSING_THRESHOLD = os.getenv('PREPROCESSING_THRESHOLD', 'true').lower() in ('1', 'true', 'yes')
PREPROCESSING_DESKEW = os.getenv('PREPROCESSING_DESKEW', 'true').lower() in ('1', 'true', 'yes')
MIN_IMAGE_DIMENSION = int(os.getenv('MIN_IMAGE_DIMENSION', '50'))
MAX_IMAGE_DIMENSION = int(os.getenv('MAX_IMAGE_DIMENSION', '10000'))


class ImagePreprocessingError(Exception):
    """Raised when preprocessing fails due to invalid/corrupt images or internal errors."""
    pass


def denoise_image(image: np.ndarray) -> np.ndarray:
    """Apply fast non-local means denoising for grayscale images.

    Args:
        image: numpy array (grayscale or color). If color, will be converted to
               grayscale for denoising then returned in same shape.

    Returns:
        denoised numpy array
    """
    start = time.time()
    try:
        if image is None:
            raise ImagePreprocessingError('Input image is None')
        # Work on grayscale for faster denoising
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        # parameters: h (filter strength) tuned modestly; templateWindowSize and searchWindowSize defaults
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
        duration = int((time.time() - start) * 1000)
        LOG.info('preprocess_denoise', extra={'duration_ms': duration})
        # return as single-channel image
        return denoised
    except Exception as e:
        LOG.exception('denoise_image_failed', exc_info=True)
        raise ImagePreprocessingError(str(e))


def threshold_image(image: np.ndarray, method: str = 'adaptive') -> np.ndarray:
    """Binarize image using adaptive thresholding or Otsu.

    Args:
        image: grayscale numpy array
        method: 'adaptive' or 'otsu'

    Returns:
        binarized numpy array
    """
    start = time.time()
    try:
        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if method == 'adaptive':
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)
        elif method == 'otsu':
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            raise ImagePreprocessingError(f'Unknown threshold method: {method}')
        duration = int((time.time() - start) * 1000)
        LOG.info('preprocess_threshold', extra={'method': method, 'duration_ms': duration})
        return thresh
    except Exception as e:
        LOG.exception('threshold_image_failed', exc_info=True)
        raise ImagePreprocessingError(str(e))


def deskew_image(image: np.ndarray) -> Tuple[np.ndarray, float]:
    """Detect and correct skew using image moments / Hough transform fallback.

    Returns:
        (rotated_image, angle_degrees)
    """
    start = time.time()
    try:
        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Use edges to detect lines
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, math.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
        angle = 0.0
        if lines is not None and len(lines) > 0:
            angles = []
            for x1, y1, x2, y2 in lines[:, 0]:
                theta = math.degrees(math.atan2(y2 - y1, x2 - x1))
                # ignore near-vertical lines
                if abs(theta) < 45:
                    angles.append(theta)
            if angles:
                angle = float(np.median(angles))
        else:
            # fallback: compute skew from moments via projection profile
            coords = np.column_stack(np.where(gray < 255))
            if coords.size == 0:
                angle = 0.0
            else:
                rect = cv2.minAreaRect(coords)
                angle = rect[-1]
                if angle < -45:
                    angle = -(90 + angle)
                else:
                    angle = -angle
        # rotate image to correct angle
        (h, w) = gray.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        duration = int((time.time() - start) * 1000)
        LOG.info('preprocess_deskew', extra={'angle': angle, 'duration_ms': duration})
        return rotated, angle
    except Exception as e:
        LOG.exception('deskew_image_failed', exc_info=True)
        raise ImagePreprocessingError(str(e))


def _validate_dimensions(w: int, h: int):
    if w < MIN_IMAGE_DIMENSION or h < MIN_IMAGE_DIMENSION:
        raise ImagePreprocessingError(f'image_too_small: {w}x{h}')
    if w > MAX_IMAGE_DIMENSION or h > MAX_IMAGE_DIMENSION:
        raise ImagePreprocessingError(f'image_too_large: {w}x{h}')


def preprocess_image(image_path: str, enable_denoise: bool = True, enable_threshold: bool = True, enable_deskew: bool = True) -> Dict[str, Any]:
    """Load an image and apply optional preprocessing steps.

    Args:
        image_path: path to the local image file
        enable_denoise: apply denoising
        enable_threshold: apply binarization
        enable_deskew: detect and correct skew

    Returns:
        dict with keys: 'original' (PIL Image), 'processed' (PIL Image), 'steps_applied' (list), 'metadata' (dict)
    """
    start_total = time.time()
    steps: List[str] = []
    try:
        pil_img = Image.open(image_path).convert('RGB')
        w, h = pil_img.size
        _validate_dimensions(w, h)
        # convert to numpy BGR for OpenCV
        np_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        processed = np_img
        metadata: Dict[str, Any] = {'original_width': w, 'original_height': h}

        if enable_denoise and PREPROCESSING_DENOISE:
            processed = denoise_image(processed)
            steps.append('denoise')
            # ensure processed is single-channel; convert back to BGR for following ops
            if len(processed.shape) == 2:
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

        if enable_threshold and PREPROCESSING_THRESHOLD:
            # threshold expects grayscale
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY) if len(processed.shape) == 3 else processed
            processed = threshold_image(gray, method='adaptive')
            steps.append('threshold')
            # convert back to BGR so later steps have 3 channels
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

        skew_angle = 0.0
        if enable_deskew and PREPROCESSING_DESKEW:
            processed, skew_angle = deskew_image(processed)
            steps.append('deskew')
            metadata['skew_angle'] = skew_angle

        # convert processed back to PIL Image RGB
        processed_pil = Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
        total_ms = int((time.time() - start_total) * 1000)
        metadata.update({'processing_time_ms': total_ms, 'steps_applied': steps})
        LOG.info('preprocess_complete', extra={'steps': steps, 'processing_time_ms': total_ms})
        return {'original': pil_img, 'processed': processed_pil, 'steps_applied': steps, 'metadata': metadata}
    except ImagePreprocessingError:
        raise
    except Exception as e:
        LOG.exception('preprocess_failed', exc_info=True)
        raise ImagePreprocessingError(str(e))
