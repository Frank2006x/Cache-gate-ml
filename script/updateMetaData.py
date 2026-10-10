import json
from pathlib import Path
   
metadata_path = Path("data/metadata/imagenet_class_index.json")
brightness_dir = Path("data/brightness/1")

# 1. Collect all synset directory names present in data/brightness/1
existing_synsets = {d.name for d in brightness_dir.iterdir() if d.is_dir()}

# 2. Load the full 1,000-class metadata
with open(metadata_path, "r") as f:
	full_metadata = json.load(f)

# 3. Filter to keep only classes existing in the brightness folder
filtered_metadata = {
	key: val for key, val in full_metadata.items()
	if val[0] in existing_synsets
}

# 4. Save the filtered JSON back to disk
with open(metadata_path, "w") as f:
	json.dump(filtered_metadata, f, indent=2)