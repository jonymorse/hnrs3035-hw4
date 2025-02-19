import json
import os

# Paths
DATA_DIR = "data"
parsed_results_path = os.path.join(DATA_DIR, f"gpt-4o-mini-{datetime.today().strftime('%Y-%m-%d')}-hw3.json")

# Load parsed results
with open(parsed_results_path, "r") as file:
    results = json.load(file)

# Compute accuracy
total = len(results)
correct = sum(1 for r in results if r["score"] is True)
accuracy = (correct / total) * 100 if total > 0 else 0

# Print results
print(f"✅ Total Questions Evaluated: {total}")
print(f"✅ Correct Answers: {correct}")
print(f"❌ Incorrect Answers: {total - correct}")
print(f"📊 Model Accuracy: {accuracy:.2f}%")
