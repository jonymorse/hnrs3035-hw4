import json
import os

# Configuration
DATA_DIR = "data"
RESULTS_FILE = os.path.join(DATA_DIR, "gpt4o-mini-eval-responses.json")
HW3_ACCURACY = 32.40  # Baseline accuracy from HW3

def calculate_model_score():
    """Calculate the total score by analyzing true/false values in responses"""
    total_responses = 0
    correct_responses = 0
    errors = 0
    
    try:
        with open(RESULTS_FILE, 'r') as f:
            for line in f:
                total_responses += 1
                response_data = json.loads(line)
                
                try:
                    # Check if response was successful
                    if response_data["response"]["status_code"] != 200:
                        print(f"❌ Error in response {response_data['custom_id']}: Status code {response_data['response']['status_code']}")
                        errors += 1
                        continue
                    
                    # Extract and parse the content which contains the score
                    content = response_data["response"]["body"]["choices"][0]["message"]["content"]
                    evaluation = json.loads(content)
                    
                    # Add to correct count if score is true
                    if evaluation["score"]:
                        correct_responses += 1
                        print(f"✓ Response {response_data['custom_id']}: Correct")
                    else:
                        print(f"✗ Response {response_data['custom_id']}: Incorrect")
                        print(f"  Explanation: {evaluation['explanation']}")
                        
                except json.JSONDecodeError as e:
                    print(f"❌ Error parsing JSON for response {response_data['custom_id']}: {str(e)}")
                    errors += 1
                except KeyError as e:
                    print(f"❌ Missing key in response {response_data['custom_id']}: {str(e)}")
                    errors += 1
                except Exception as e:
                    print(f"❌ Unexpected error processing response {response_data['custom_id']}: {str(e)}")
                    errors += 1
                    
    except FileNotFoundError:
        print(f"❌ Results file not found: {RESULTS_FILE}")
        return
    
    # Calculate statistics
    valid_responses = total_responses - errors
    current_accuracy = (correct_responses / valid_responses * 100) if valid_responses > 0 else 0
    
    # Calculate improvement metrics
    accuracy_improvement = current_accuracy - HW3_ACCURACY
    percent_improvement = (accuracy_improvement / HW3_ACCURACY) * 100
    
    print(f"""
📊 Model Score Summary:
   - Total Responses: {total_responses}
   - Correct Responses: {correct_responses}
   - Incorrect Responses: {valid_responses - correct_responses}
   - Errors: {errors}
   - Current Accuracy: {current_accuracy:.2f}%

📈 Comparison with HW3:
   - HW3 Baseline: {HW3_ACCURACY}%
   - Current Score: {current_accuracy:.2f}%
   - Absolute Improvement: {accuracy_improvement:+.2f}%
   - Relative Improvement: {percent_improvement:+.2f}%
""")

    # Print performance comparison message
    if accuracy_improvement > 0:
        print("✨ Performance improved compared to HW3!")
    elif accuracy_improvement < 0:
        print("📉 Performance decreased compared to HW3.")
    else:
        print("➡️ Performance remained the same as HW3.")

if __name__ == "__main__":
    calculate_model_score()