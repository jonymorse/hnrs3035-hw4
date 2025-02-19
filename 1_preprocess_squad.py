import json
import os

# Ensure data directory exists
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# Load SQuAD2.0 Dataset
squad_path = os.path.join(data_dir, "dev-v2.0.json")

with open(squad_path, "r") as file:
    squad_data = json.load(file)

questions = []
count = 0  # Keep track of the number of extracted questions

for article in squad_data["data"]:
    for paragraph in article["paragraphs"]:
        for qa in paragraph["qas"]:
            if not qa["is_impossible"]:  # Skip impossible questions
                questions.append({
                    "question": qa["question"],
                    "answers": [ans["text"] for ans in qa["answers"]]
                })
                count += 1
            if count >= 500:
                break  # Stop once we have 500 questions
        if count >= 500:
            break
    if count >= 500:
        break

# Save processed questions
filtered_questions_path = os.path.join(data_dir, "filtered_questions.json")

with open(filtered_questions_path, "w") as file:
    json.dump(questions, file, indent=2)

print(f"Extracted exactly {len(questions)} questions ✅")
