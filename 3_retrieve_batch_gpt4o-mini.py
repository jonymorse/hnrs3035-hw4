import openai
import os
import json
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY not found. Please check your .env file.")

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Define batch ID (replace with actual ID)
BATCH_ID = "batch_67ad3d5ae3248190ac851067b7eb8727"

# Retrieve batch details
batch_details = client.batches.retrieve(BATCH_ID)
output_file_id = batch_details.output_file_id

if output_file_id:
    # Retrieve the output file
    file_response = client.files.content(output_file_id)
    file_contents = file_response.text  # ✅ Fix: Remove parentheses

    # Convert JSONL to JSON
    output_json = []
    for line in file_contents.split("\n"):
        if line.strip():
            output_json.append(json.loads(line))

    # Save as a JSON file
    output_path = "data/gpt4o-mini-responses.json"
    with open(output_path, "w") as file:
        json.dump(output_json, file, indent=2)

    print(f"✅ GPT-4o-mini responses saved to {output_path}")

else:
    print("❌ No output file found. The batch may not have completed successfully.")
