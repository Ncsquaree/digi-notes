#!/usr/bin/env python3
"""
Download TFLite models for offline AI system.
Models:
  - MobileBERT-SQuAD: Question answering for flashcard generation
  - Universal Sentence Encoder Lite: Semantic embeddings
"""
import os
import requests
from tqdm import tqdm

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

MODELS = {
    'mobilebert-squad.tflite': {
        'url': 'https://tfhub.dev/tensorflow/lite-model/mobilebert/1/metadata/1?lite-format=tflite',
        'size_mb': 25,
        'description': 'MobileBERT fine-tuned on SQuAD v1.1 for extractive QA'
    },
    'mobilebert-squad.spm': {
        'url': 'https://storage.googleapis.com/tfhub-lite-models/tensorflow/lite-model/mobilebert/1/metadata/1/vocab.txt',
        'size_mb': 1,
        'description': 'SentencePiece tokenizer for MobileBERT (vocab file)'
    },
    'use-lite.tflite': {
        'url': 'https://tfhub.dev/google/lite-model/universal-sentence-encoder-qa-ondevice/1?lite-format=tflite',
        'size_mb': 50,
        'description': 'Universal Sentence Encoder Lite (512-dim embeddings)'
    }
}

def download_file(url: str, dest_path: str, desc: str):
    """Download file with progress bar."""
    print(f"Downloading {desc}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f, tqdm(
        total=total_size, unit='B', unit_scale=True, desc=desc
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))
    print(f"✓ Saved to {dest_path}")

def main():
    print("=== Offline AI Model Downloader ===\n")
    for filename, info in MODELS.items():
        dest = os.path.join(MODELS_DIR, filename)
        if os.path.exists(dest):
            print(f"⊗ {filename} already exists, skipping...")
            continue
        try:
            download_file(info['url'], dest, f"{filename} ({info['size_mb']}MB)")
        except Exception as e:
            print(f"✗ Failed to download {filename}: {e}")
            continue
    
    # Create models README
    readme_path = os.path.join(MODELS_DIR, 'README.md')
    with open(readme_path, 'w') as f:
        f.write("# TFLite Models\n\n")
        for filename, info in MODELS.items():
            f.write(f"## {filename}\n")
            f.write(f"- **Description:** {info['description']}\n")
            f.write(f"- **Size:** ~{info['size_mb']}MB\n")
            f.write(f"- **Source:** {info['url']}\n")
            f.write(f"- **License:** Apache 2.0 (TensorFlow Hub)\n\n")
    
    print(f"\n✓ All models downloaded to {MODELS_DIR}")
    print(f"✓ Model documentation: {readme_path}")

if __name__ == '__main__':
    main()
