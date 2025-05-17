# rag_chatbot.py

import os
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ==== Configuration ====
FALCON_BASE = "tiiuae/Falcon3-1B-Instruct"
ADAPTER_PATH = "model/falcon-lora-finetuned"
VECTOR_DB_DIR = "data/embeddings/rag_db"  # must match your vector DB folder
COLLECTION_NAME = "hotel_qa"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OFFLOAD_DIR = "offload"  # folder to store offloaded layers

# ==== Initialize RAG Components ====
@st.cache_resource(show_spinner=False)
def init_rag():
    st.write("🔄 Loading tokenizer and Falcon model with LoRA offloading...")

    os.makedirs(OFFLOAD_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(FALCON_BASE, trust_remote_code=True)

    # Load base model with minimal memory usage (CPU first)
    base_model = AutoModelForCausalLM.from_pretrained(
        FALCON_BASE,
        device_map="auto",                  # auto-dispatch layers to GPU/CPU
        torch_dtype=torch.float16,
        trust_remote_code=True,
        offload_folder=OFFLOAD_DIR         # <=== important for Accelerate
    )

    # ✅ Load LoRA adapter with matching offload config
    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
        device_map="auto",
        offload_folder=OFFLOAD_DIR,
        torch_dtype=torch.float16
    )

    model.eval()

    st.write("🧠 Loading ChromaDB collection...")
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL_NAME)
    client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)

    return tokenizer, model, collection


# ==== Context Retrieval ====
def retrieve_context(query, collection, k=3):
    results = collection.query(query_texts=[query], n_results=k)
    docs = results.get("documents", [[]])[0]
    return "\n".join(docs)

# ==== Answer Generation ====
# def generate_answer(query, context, model, tokenizer):
#     prompt = f"""
# Answer the question based only on the context below.
# If the answer is not in the context, respond with: "Sorry, I don't know."
# If the context includes a list seperated by semicolons (;), present each item as a separate bullet point
# Do not add any extra information or repeat the question. 

# Context:
# {context}

# Question: {query}
# Answer:
# """
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     outputs = model.generate(**inputs, max_new_tokens=150)
#     decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return decoded.split("Answer:")[-1].strip()

def generate_answer(query, context, model, tokenizer):
    prompt = f"""
You are a helpful assistant.

Answer the following question using *only* the information in the provided context.  
- If the context includes a list separated by semicolons (;), present each item as a separate bullet point.  
- Do not add any extra information or repeat the question.  
- If the answer is not found in the context, respond with:  Sorry, I don't know.

Context:
\"\"\"{context}\"\"\"

Question: {query}

Answer (in bullet points only, if applicable):
- """
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False, num_return_sequences=1)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the part after "Answer..." to avoid duplicate answer blocks
    if "Answer (in bullet points only, if applicable):" in decoded:
        answer = decoded.split("Answer (in bullet points only, if applicable):")[-1].strip()
    else:
        answer = decoded.strip()
    
    # Ensure single bullet list
    answer_lines = answer.splitlines()
    unique_lines = []
    seen = set()
    for line in answer_lines:
        line = line.strip()
        if line and line not in seen:
            seen.add(line)
            unique_lines.append(line)
    
    return "\n".join(unique_lines)

# ==== Streamlit UI ====
def main():
    st.set_page_config(page_title="Taj Hotel RAG Chatbot", layout="centered")
    st.title("🏨 Taj Hotel Chatbot (RAG + LoRA)")
    st.markdown("Ask me anything about **The Taj Mahal Palace, Mumbai**.")

    query = st.text_input("❓ Enter your question:", placeholder="e.g., What are the dining options?")

    if query:
        tokenizer, model, collection = init_rag()
        context = retrieve_context(query, collection)
        answer = generate_answer(query, context, model, tokenizer)
        st.markdown("#### 💬 Answer:")
        st.success(answer)

if __name__ == "__main__":
    main()
