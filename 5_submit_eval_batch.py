import openai
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
batch_input_path = os.path.join(DATA_DIR, "eval_batch_input.jsonl")

# Step 1: Upload batch input file
print("⏳ Uploading batch input file...")
uploaded_file = client.files.create(
    file=open(batch_input_path, "rb"),
    purpose="batch"
)
file_id = uploaded_file.id
print(f"✅ File uploaded successfully! File ID: {file_id}")

# Step 2: Submit batch job
print("⏳ Submitting batch job...")
batch = client.batches.create(
    input_file_id=file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h"
)
batch_id = batch.id
print(f"✅ Batch job submitted! Batch ID: {batch_id}")

# Step 3: Monitor batch status
print("⏳ Waiting for batch processing...")
while True:
    batch_status = client.batches.retrieve(batch_id)
    print(f"🔄 Batch Status: {batch_status.status}")

    if batch_status.status in ["completed", "failed", "expired", "cancelled"]:
        break  # Exit loop when batch processing is complete

    time.sleep(30)  # Wait 30 seconds before checking again

# Step 4: Final result
if batch_status.status == "completed":
    print(f"✅ Batch completed successfully! Output file ID: {batch_status.output_file_id}")
elif batch_status.status == "failed":
    print("❌ Batch processing failed.")
elif batch_status.status == "expired":
    print("⚠️ Batch expired before completion.")
elif batch_status.status == "cancelled":
    print("❌ Batch was cancelled.")

print("🚀 Done!")
