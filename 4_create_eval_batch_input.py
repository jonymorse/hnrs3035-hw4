import json
import os
from datetime import datetime
import tiktoken

# Paths
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
questions_path = os.path.join(DATA_DIR, "filtered_questions.json")
responses_path = os.path.join(DATA_DIR, "gpt4o-mini-responses.json")
batch_input_path = os.path.join(DATA_DIR, "eval_batch_input.jsonl")

# Load Questions
with open(questions_path, "r") as file:
    questions = json.load(file)

# Load GPT-4o-mini Responses
with open(responses_path, "r") as file:
    responses = json.load(file)

# Token Calculation Function
def num_tokens_from_string(string: str, model_name: str) -> int:
    """Returns the number of tokens used by a string for a specific model."""
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Model name
model_name = "gpt-4o"  # ✅ Use GPT-4o-mini for better token limits

# Prepare batch input file
MAX_REQUESTS = 500  # ✅ Reduce batch size to avoid token limits
total_tokens = 0
with open(batch_input_path, "w") as file:
    for idx, (question, response_data) in enumerate(zip(questions, responses)):
        if idx >= MAX_REQUESTS:
            break  # ✅ Limit batch size

        system_message = (
            "You are a teacher tasked with determining whether a student's answer to a question was correct, "
            "based on a set of possible correct answers. You must only use the provided possible correct answers "
            "to determine if the student's response was correct."
        )
        user_message = (
            f"Question: {question['question']}\n\n"
            f"Student’s Response: {response_data['response']}\n\n"
            f"Possible Correct Answers:\n{question['answers']}\n\n"
            "Your response should only be a valid JSON as shown below:\n"
            "{\n  \"explanation\": \"A short explanation of why the student’s answer was correct or incorrect.\",\n"
            "  \"score\": true if the student’s answer was correct, false if it was incorrect\n}"
        )

        # Calculate tokens for each message
        system_tokens = num_tokens_from_string(system_message, model_name)
        user_tokens = num_tokens_from_string(user_message, model_name)

        # Total tokens for this request
        request_tokens = system_tokens + user_tokens + 50  # ✅ Reduce output token limit
        total_tokens += request_tokens

        eval_prompt = {
            "custom_id": f"eval-{idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                "temperature": 0.0,  # ✅ Deterministic evaluation
                "max_tokens": 200,  # ✅ Reduce max tokens
                "top_p": 1.0,
                "response_format": {"type": "json_object"}  # ✅ Fixed format
            }
        }
        file.write(json.dumps(eval_prompt) + "\n")

print(f"✅ Evaluation batch input file created: {batch_input_path}")
print(f"Total tokens for all requests: {total_tokens}")
