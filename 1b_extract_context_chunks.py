import json

# Load dataset
with open("data/dev-v2.0.json", "r") as f:
    squad_data = json.load(f)

# Extract contexts using list comprehension
contexts = [paragraph["context"] for article in squad_data["data"] for paragraph in article["paragraphs"]]

# Save extracted contexts
with open("data/contexts.json", "w") as f:
    json.dump(contexts, f, indent=4)

print(f"Extracted {len(contexts)} context chunks and saved them to 'data/contexts.json'.")
