import requests
import os
from tqdm import tqdm

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# URL for SQuAD2.0 Dev Set
SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
# Alternative URL (uncomment if needed)
# SQUAD_URL = "https://huggingface.co/datasets/GEM/squad_v2/blob/main/squad_data/dev-v2.0.json"

# Display message before download starts
print("Downloading SQuAD2.0 Dev Set...")

# Stream the response to show progress
response = requests.get(SQUAD_URL, stream=True)
total_size = int(response.headers.get("content-length", 0))  # Get total file size
chunk_size = 1024  # Read in 1 KB chunks

with open("data/dev-v2.0.json", "wb") as file, tqdm(
    desc="Downloading",
    total=total_size,
    unit='B',
    unit_scale=True,
    unit_divisor=1024,
) as bar:
    for chunk in response.iter_content(chunk_size=chunk_size):
        if chunk:
            file.write(chunk)
            bar.update(len(chunk))

print("\nDownload complete.")
