import openai
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_JOMO")

if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY not found. Please check your .env file.")

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Batch ID (Replace with your actual Batch ID)
#BATCH_ID = "batch_67b7b1cd76f481909d0aef65d661c4e5"  # ✅ Replace with your actual Batch ID
BATCH_ID = "batch_67b7ac36df308190aab2a321fe093ef9"
# Retrieve batch details
print("⏳ Retrieving batch details, please wait...")

try:
    batch_details = client.batches.retrieve(BATCH_ID)
    output_file_id = batch_details.output_file_id

    if not output_file_id:
        raise ValueError("❌ No output file found. The batch may not have completed successfully.")

    print(f"✅ Output File ID: {output_file_id}")

    # Retrieve the output file
    print("⏳ Retrieving batch results, please wait...")
    file_response = client.files.content(output_file_id)
    file_contents = file_response.text

    # Save the results to a JSONL file (line-by-line JSON format)
    output_path = "data/eval_batch_results.jsonl"
    with open(output_path, "w") as file:
        file.write(file_contents)

    print(f"✅ Batch results saved to {output_path}")

except Exception as e:
    print(f"❌ Error retrieving batch results: {e}")
