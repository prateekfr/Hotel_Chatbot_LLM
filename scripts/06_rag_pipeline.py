import os
import json
import shutil
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ==== Config ====
DATA_PATH = "data/processed/processed_dataset.json"
VECTOR_DB_DIR = "data/embeddings/rag_db"
COLLECTION_NAME = "hotel_qa"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ==== Load Dataset ====
def load_qa_pairs(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    parsed = [item for item in data if "question" in item and "answer" in item]
    return pd.DataFrame(parsed)

# ==== Build Chroma Vector Store ====
def build_vector_db(df):
    print("Initializing Chroma DB...")
    if os.path.exists(VECTOR_DB_DIR):
        print(f"Deleting existing DB at {VECTOR_DB_DIR}...")
        shutil.rmtree(VECTOR_DB_DIR)

    client = PersistentClient(path=VECTOR_DB_DIR)

    print("Loading embedding model...")
    embedder = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL_NAME)

    print("Creating collection...")
    collection = client.create_collection(name=COLLECTION_NAME, embedding_function=embedder)

    print("📨 Adding documents to vector store...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        collection.add(
            documents=[row["answer"]],
            metadatas=[{"question": row["question"]}],
            ids=[f"q_{idx}"]
        )

    print(f" Vector DB created with {len(df)} documents → {VECTOR_DB_DIR}/")

# ==== Main Entry ====
if __name__ == "__main__":
    df = load_qa_pairs(DATA_PATH)
    build_vector_db(df)
