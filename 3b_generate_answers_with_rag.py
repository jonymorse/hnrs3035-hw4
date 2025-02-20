import json
import os

# Paths
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
batch_input_path = os.path.join(DATA_DIR, "gpt_batch_requests.jsonl")

# Load retrieved contexts for each question
with open("data/rag_inputs.json", "r") as f:
    rag_inputs = json.load(f)

print(f"🔍 Loaded {len(rag_inputs)} questions with retrieved contexts.")

# Prepare batch requests for OpenAI
PROMPT_TEMPLATE = """You are an AI assistant that answers questions concisely and accurately.
Below are multiple retrieved context passages relevant to the question:

Contexts:
{contexts}

Question: {question}

Provide a well-structured and concise answer based on the contexts above.
"""

with open(batch_input_path, "w") as file:
    for idx, item in enumerate(rag_inputs):
        retrieved_contexts = "\n\n".join(item["retrieved_contexts"])
        request_data = {
            "custom_id": f"request-{idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that answers questions concisely and accurately."},
                    {"role": "user", "content": PROMPT_TEMPLATE.format(question=item["question"], contexts=retrieved_contexts)}
                ],
                "temperature": 0.5,
                "max_tokens": 100,
                "top_p": 0.9,
                "store": True  # Enables OpenAI optimization
            }
        }
        file.write(json.dumps(request_data) + "\n")

print(f"✅ Batch input file created: {batch_input_path}")
