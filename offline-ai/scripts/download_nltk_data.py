#!/usr/bin/env python3
"""
Download required NLTK data for offline AI system.

Downloads:
  - punkt: Sentence tokenizer models
  - averaged_perceptron_tagger: Part-of-speech tagger (optional for Phase 4)
"""
import nltk
import sys

REQUIRED_DATA = {
    'tokenizers/punkt': 'Sentence tokenizer (required)',
    'taggers/averaged_perceptron_tagger': 'POS tagger (optional for Phase 4)',
}

def download_nltk_data():
    """Download NLTK data packages."""
    print("=== NLTK Data Downloader ===\n")
    
    failed = []
    for data_name, description in REQUIRED_DATA.items():
        try:
            nltk.data.find(data_name)
            print(f"✓ {data_name}: Already available - {description}")
        except LookupError:
            print(f"⊗ {data_name}: Not found, downloading...")
            try:
                # Extract package name from path (e.g., 'tokenizers/punkt' → 'punkt')
                package = data_name.split('/')[-1]
                nltk.download(package, quiet=True)
                print(f"✓ {data_name}: Downloaded successfully")
            except Exception as e:
                print(f"✗ {data_name}: Download failed - {e}")
                failed.append(data_name)
    
    if failed:
        print(f"\n✗ Failed to download: {', '.join(failed)}")
        print("  Try manual download with: python -m nltk.downloader punkt")
        return False
    
    print("\n✓ All required NLTK data downloaded successfully")
    return True

if __name__ == '__main__':
    success = download_nltk_data()
    sys.exit(0 if success else 1)
