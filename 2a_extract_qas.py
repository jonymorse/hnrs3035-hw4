import json

# Load dataset from file
with open("data/dev-v2.0.json", "r") as f:
    squad_data = json.load(f)

# Extract first 500 valid questions and answers
questions_answers = []
count = 0

for article in squad_data["data"]:
    for paragraph in article["paragraphs"]:
        for qa in paragraph["qas"]:
            if not qa["is_impossible"]:  # Skip unanswerable questions
                question = qa["question"]
                answer = qa["answers"][0]["text"] if qa["answers"] else "N/A"
                questions_answers.append({"question": question, "answer": answer})
                count += 1
                if count == 500:  # Stop when we reach 500 valid questions
                    break
        if count == 500:
            break
    if count == 500:
        break

# Save extracted questions and answers
with open("data/questions_answers.json", "w") as f:
    json.dump(questions_answers, f, indent=4)

print(f"✅ Extracted {len(questions_answers)} valid questions and answers and saved them to 'data/questions_answers.json'.")
