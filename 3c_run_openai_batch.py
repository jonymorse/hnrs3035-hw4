import openai
import json
import os
import time
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY not found. Please check your .env file.")

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Paths
DATA_DIR = "data"
batch_input_path = os.path.join(DATA_DIR, "gpt_batch_requests.jsonl")
batch_output_path = os.path.join(DATA_DIR, "generated_answers.json")

# Step 1: Upload File to OpenAI
uploaded_file = client.files.create(
    file=open(batch_input_path, "rb"),
    purpose="batch"
)
file_id = uploaded_file.id
print(f"✅ File uploaded successfully! File ID: {file_id}")

# Step 2: Submit Batch Job
batch = client.batches.create(
    input_file_id=file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h"
)
batch_id = batch.id
print(f"✅ Batch job submitted! Batch ID: {batch_id}")

# Step 3: Monitor Batch Job
print("⏳ Waiting for batch completion...")
while True:
    batch_status = client.batches.retrieve(batch_id)
    print(f"🔄 Batch Status: {batch_status.status}")
    
    if batch_status.status in ["completed", "failed", "expired", "cancelled"]:
        break  # Exit loop when batch processing is complete
    
    time.sleep(60)  # Wait 1 minute before checking again

# Step 4: Retrieve Results
if batch_status.status == "completed":
    output_file_id = batch_status.output_file_id
    file_response = client.files.content(output_file_id)
    file_contents = file_response.text()

    with open(batch_output_path, "w") as file:
        file.write(file_contents)
    
    print(f"✅ Results saved to {batch_output_path}")
else:
    print(f"❌ Batch failed with status: {batch_status.status}")
