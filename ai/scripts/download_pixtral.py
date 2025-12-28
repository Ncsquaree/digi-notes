"""Download the Pixtral model locally using transformers.

Usage:
  python scripts/download_pixtral.py

Honors environment variables:
- PIXTRAL_MODEL (default: mistral-community/pixtral-12b)
- TRANSFORMERS_CACHE (default: ./models/pixtral-12b)

Downloads processor and model weights so offline inference can run without API calls.
"""
import os
from pathlib import Path

from transformers import AutoProcessor, AutoModelForCausalLM


def main():
    model_id = os.getenv('PIXTRAL_MODEL', 'mistral-community/pixtral-12b')
    cache_dir = os.getenv('TRANSFORMERS_CACHE') or str(Path(__file__).resolve().parents[1] / 'models' / 'pixtral-12b')
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Pixtral model '{model_id}' to cache: {cache_path}")
    proc = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
    # Touch attributes to avoid lint warnings
    _ = proc, model
    print('Download complete.')


if __name__ == '__main__':
    main()
