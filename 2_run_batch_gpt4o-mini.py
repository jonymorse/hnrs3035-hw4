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
os.makedirs(DATA_DIR, exist_ok=True)
questions_path = os.path.join(DATA_DIR, "filtered_questions.json")
batch_input_path = os.path.join(DATA_DIR, "batch_input.jsonl")
batch_output_path = os.path.join(DATA_DIR, "batch_output.json")

# Step 1: Load Questions
with open(questions_path, "r") as file:
    questions = json.load(file)

# Step 2: Create JSONL Batch Input File
PROMPT_TEMPLATE = """You are an AI assistant that answers questions concisely and accurately.
Answer the following question:
Question: {question}
"""

with open(batch_input_path, "w") as file:
    for idx, item in enumerate(questions):
        request_data = {
            "custom_id": f"request-{idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that answers questions concisely and accurately."},
                    {"role": "user", "content": PROMPT_TEMPLATE.format(question=item["question"])}
                ],
                "temperature": 0.5,
                "max_tokens": 100,
                "top_p": 0.9,
                "store": True  # Enables OpenAI optimization
            }
        }
        file.write(json.dumps(request_data) + "\n")

print(f"✅ Batch input file created: {batch_input_path}")

# Step 3: Upload File to OpenAI
uploaded_file = client.files.create(
    file=open(batch_input_path, "rb"),
    purpose="batch"
)
file_id = uploaded_file.id
print(f"✅ File uploaded successfully! File ID: {file_id}")

# Step 4: Submit Batch Job
batch = client.batches.create(
    input_file_id=file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h"
)
batch_id = batch.id
print(f"✅ Batch job submitted! Batch ID: {batch_id}")

# Step 5: Check Batch Status Periodically
print("⏳ Waiting for batch completion...")
while True:
    batch_status = client.batches.retrieve(batch_id)
    print(f"🔄 Batch Status: {batch_status.status}")
    
    if batch_status.status in ["completed", "failed", "expired", "cancelled"]:
        break  # Exit loop when batch processing is complete
    
    time.sleep(60)  # Wait 1 minute before checking again

# Step 6: Retrieve Batch Results
if batch_status.status == "completed":
    output_file_id = batch_status.output_file_id
    file_response = client.files.content(output_file_id)
    file_contents = file_response.text()

    with open(batch_output_path, "w") as file:
        file.write(file_contents)
    
    print(f"✅ Results saved to {batch_output_path}")
else:
    print(f"❌ Batch failed with status: {batch_status.status}")

# Step 7: Answer Assignment Question 1 (What prompt did you use?)
print("\n📌 **Assignment Answer - Question 1:**")
print(f"The prompt used for GPT-4o-mini was:\n\n{PROMPT_TEMPLATE}")
