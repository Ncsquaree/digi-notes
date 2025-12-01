"""Image / text preprocessing utilities (placeholder)."""
from PIL import Image

def preprocess_image(path: str, out_path: str):
	# Very small placeholder: open and save as RGB to normalize
	img = Image.open(path).convert('RGB')
	img.save(out_path)
	return out_path

