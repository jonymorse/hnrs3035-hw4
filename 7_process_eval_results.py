import json
import os

# Paths
DATA_DIR = "data"
eval_results_path = os.path.join(DATA_DIR, "eval_batch_results_0.jsonl")

# Load JSONL file
eval_results = []
with open(eval_results_path, "r") as file:
    for line in file:
        try:
            eval_results.append(json.loads(line))
        except json.JSONDecodeError:
            print("❌ Error parsing a line in JSONL file")

# Print the number of records loaded
print(f"✅ Loaded {len(eval_results)} evaluation results")

# Initialize counters
total_questions = len(eval_results)
correct_answers = 0
total_prompt_tokens = 0
total_completion_tokens = 0
failed_responses = []

# Process each evaluation result
for result in eval_results:
    custom_id = result.get("custom_id", "Unknown_ID")
    response_data = result.get("response", {})

    if response_data.get("status_code") == 200:  # Ensure it's a successful response
        body = response_data.get("body", {})
        choices = body.get("choices", [])

        if choices:
            message = choices[0].get("message", {})

            if "content" in message:
                try:
                    # Parse JSON response
                    parsed_response = json.loads(message["content"])
                    explanation = parsed_response.get("explanation", "")
                    score = parsed_response.get("score", False)  # True/False

                    if score:
                        correct_answers += 1

                except json.JSONDecodeError:
                    print(f"⚠️ JSON parsing error for response: {custom_id}")
                    failed_responses.append({"custom_id": custom_id, "reason": "JSON decoding error", "raw_response": message})
            else:
                print(f"⚠️ Missing content in response: {custom_id}")
                failed_responses.append({"custom_id": custom_id, "reason": "Missing content", "raw_response": message})

        else:
            print(f"⚠️ No choices found in response: {custom_id}")
            failed_responses.append({"custom_id": custom_id, "reason": "No choices", "raw_response": body})

        # Track token usage
        usage = body.get("usage", {})
        total_prompt_tokens += usage.get("prompt_tokens", 0)
        total_completion_tokens += usage.get("completion_tokens", 0)

    else:
        error_message = response_data.get("body", {}).get("error", {}).get("message", "Unknown error")
        print(f"❌ Failed API response: {custom_id} | Error: {error_message}")
        failed_responses.append({"custom_id": custom_id, "reason": "API failure", "error": error_message})

# Calculate accuracy
accuracy = (correct_answers / total_questions) * 100 if total_questions > 0 else 0

# Print results
print("\n📊 **Evaluation Insights**")
print(f"✅ Total Questions Evaluated: {total_questions}")
print(f"✅ Correct Answers: {correct_answers}")
print(f"✅ Accuracy: {accuracy:.2f}%")
print(f"✅ Total Prompt Tokens Used: {total_prompt_tokens}")
print(f"✅ Total Completion Tokens Used: {total_completion_tokens}")
print(f"✅ Total Tokens Used: {total_prompt_tokens + total_completion_tokens}")

# Extract failed response IDs
failed_ids = [resp["custom_id"] for resp in failed_responses]
print("\n❌ Failed Response IDs (First 10 shown):")
print(failed_ids[:10])

# Log any failures
if failed_responses:
    print(f"❌ Failed Responses: {len(failed_responses)}")
    print(f"❌ IDs of Failed Responses: {failed_ids}")

    # Print details of first 5 failed responses for debugging
    print("\n🚨 **Example Failed Responses (First 5 Shown)**")
    for i, failed in enumerate(failed_responses[:5]):  # Print first 5 failed cases
        print(f"\n🚨 **Failed Response {i+1}:**")
        print(json.dumps(failed, indent=2))
        print("-" * 80)

