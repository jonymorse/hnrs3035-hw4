import chromadb

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="squad_contexts")

# Get stored contexts
results = collection.get()  # Retrieves all stored documents (limit may apply)

if "metadatas" in results and results["metadatas"]:
    print("🔎 Sample stored contexts in ChromaDB:")
    for i, item in enumerate(results["metadatas"][:3]):  # Print first 3 results
        print(f"\n🔹 Context {i+1}:")
        print(item["text"])
else:
    print("⚠️ No data found in ChromaDB. Make sure 1c_store_contexts.py ran successfully!")
