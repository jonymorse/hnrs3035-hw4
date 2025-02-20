import json
import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Get OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OpenAI API key is missing! Ensure it is set in the '.env' file.")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)
embed_model = "text-embedding-3-small"

# Load context chunks
with open("data/contexts.json", "r") as file:
    contexts = json.load(file)

print(f"Loaded {len(contexts)} context chunks from 'data/contexts.json'.")

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="squad_contexts")

print("ChromaDB initialized and collection created.")

def get_embedding(text):
    """Fetch an embedding for a given text using OpenAI."""
    response = client.embeddings.create(model=embed_model, input=text)
    return response.data[0].embedding

# Store Contexts and Embeddings in ChromaDB
for i, context in enumerate(contexts):
    embedding = get_embedding(context)  # Generate OpenAI embedding
    collection.add(
        ids=[f"context_{i}"],  # Unique ID
        embeddings=[embedding],  # Store embedding
        metadatas=[{"text": context}]  # Store original context text
    )

print("✅ Successfully stored all context chunks in ChromaDB!")
