import json
import os
from datetime import datetime
import argparse

# Paths
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
eval_batch_input_path = os.path.join(DATA_DIR, "gpt4o-mini-eval-requests.jsonl")

def load_data():
    """Load generated responses and ground truth answers"""
    with open(os.path.join(DATA_DIR, "gpt4o-mini-responses.json"), "r") as f:
        generated_responses = json.load(f)
    with open(os.path.join(DATA_DIR, "questions_answers.json"), "r") as f:
        ground_truth = json.load(f)
    return generated_responses, ground_truth

def create_evaluation_prompt(question: str, student_response: str, correct_answers: str) -> str:
    """Create evaluation prompt with proper formatting"""
    return f"""You are a teacher tasked with determining whether a student's answer to a question was correct,
based on a set of possible correct answers. You must only use the provided possible correct answers
to determine if the student's response was correct.
Question: {question}
Student's Response: {student_response}
Possible Correct Answers:
{correct_answers}
Your response should only be a valid Json as shown below:
{{
"explanation" (str): A short explanation of why the student's answer was correct or incorrect.,
"score" (bool): true if the student's answer was correct, false if it was incorrect
}}
Your response: """

def prepare_evaluation_requests(batch_size=None):
    """
    Prepare evaluation requests and save to JSONL file
    
    Args:
        batch_size (int, optional): Number of evaluations to generate. If None, processes all responses.
    """
    generated_responses, ground_truth = load_data()
    
    # Limit the number of responses if batch_size is specified
    if batch_size is not None:
        batch_size = min(batch_size, len(generated_responses))
        generated_responses = generated_responses[:batch_size]
    
    # Create mapping of question ID to ground truth
    question_map = {f"request-{i}": q for i, q in enumerate(ground_truth)}
    print(f"🔍 Processing {len(generated_responses)} responses for evaluation...")
    
    with open(eval_batch_input_path, "w") as file:
        for idx, entry in enumerate(generated_responses):
            custom_id = entry["custom_id"]
            response_body = entry.get("response", {}).get("body", {})
            choices = response_body.get("choices", [])
            
            # Extract generated answer
            if choices:
                generated_answer = choices[0].get("message", {}).get("content", "No response generated.")
            else:
                generated_answer = "No response generated."
                
            # Get ground truth answer
            question_data = question_map.get(custom_id, {})
            question = question_data.get("question", "No question available.")
            ground_truth_answer = question_data.get("answer", "No ground truth available.")
            
            # Create evaluation request
            request_data = {
                "custom_id": f"eval-{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini-2024-07-18",
                    "messages": [
                        {"role": "system", "content": "You are a strict evaluator of student responses."},
                        {"role": "user", "content": create_evaluation_prompt(
                            question=question,
                            student_response=generated_answer,
                            correct_answers=ground_truth_answer
                        )}
                    ],
                    "temperature": 0,
                    "max_tokens": 200,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "evaluation_response",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "explanation": {
                                        "type": "string",
                                        "description": "A short explanation of why the answer was correct or incorrect"
                                    },
                                    "score": {
                                        "type": "boolean",
                                        "description": "True if the answer is correct, False if incorrect"
                                    }
                                },
                                "required": ["explanation", "score"],
                                "additionalProperties": False
                            },
                            "strict": True
                        }
                    }
                }
            }
            file.write(json.dumps(request_data) + "\n")
    
    print(f"✅ Evaluation requests saved to: {eval_batch_input_path}")

def main():
    parser = argparse.ArgumentParser(description='Prepare evaluation requests for GPT-4o-mini')
    parser.add_argument('--batch-size', type=int, help='Number of evaluations to generate (default: all)', default=None)
    args = parser.parse_args()
    
    prepare_evaluation_requests(args.batch_size)

if __name__ == "__main__":
    main()