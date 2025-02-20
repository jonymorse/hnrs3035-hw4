import json
import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OpenAI API key is missing! Ensure it is set in the '.env' file.")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)
embed_model = "text-embedding-3-small"

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="squad_contexts")

# Load extracted questions
with open("data/questions_answers.json", "r") as f:
    questions_answers = json.load(f)

print(f"🔍 Loaded {len(questions_answers)} questions from 'data/questions_answers.json'.")

def get_embedding(text):
    """Fetch an embedding for a given text using OpenAI."""
    response = client.embeddings.create(model=embed_model, input=text)
    return response.data[0].embedding

# Retrieve top 5 contexts for each question
rag_inputs = []

for i, qa in enumerate(questions_answers):
    question = qa["question"]

    # Get embedding for the question
    query_embedding = get_embedding(question)

    # Query ChromaDB for the top 5 most relevant context chunks
    results = collection.query(query_embeddings=[query_embedding], n_results=5)

    # Extract retrieved contexts
    retrieved_contexts = [meta["text"] for meta in results["metadatas"][0]]

    # Store data for RAG pipeline
    rag_inputs.append({
        "question": question,
        "retrieved_contexts": retrieved_contexts
    })

    # Print progress every 50 questions
    if (i + 1) % 50 == 0:
        print(f"✅ Processed {i + 1} questions...")

# Save the data for RAG pipeline
with open("data/rag_inputs.json", "w") as f:
    json.dump(rag_inputs, f, indent=4)

print("🎯 Successfully retrieved top 5 contexts for each question and saved them to 'data/rag_inputs.json'.")
