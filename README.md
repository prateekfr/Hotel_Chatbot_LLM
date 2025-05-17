# 🏨 Hotel Chatbot – LLM Finetuned

## 📌 Project Overview

This project aims to build a **domain-specific chatbot** that answers questions **only** based on knowledge scraped from a specific hotel chain’s website (e.g., Taj Mahal Palace,Mumbai). The pipeline includes:
- **Web scraping**
- **Knowledge base creation**
- **Fine-tuning a Falcon-1B-Instruct model using QLoRA**
- **Streamlit-based interactive chat interface**

---

## DEMO VIDEO
- GDrive Link :- https://drive.google.com/file/d/1FItfaTaidGFfB6VOh6LRwH_RCuo3tc9G/view?usp=sharing

---


## 🚀 Features
- ⚙️ **Fine-tuned LLM (Falcon1B-Instruct)** with LoRA adapters and Quantizations
- 📄 Strictly domain-bound responses via fine-tuning 
- 💬 Easy-to-use chatbot UI with Streamlit
- 📚 Structured QA training using `.jsonl` knowledge base
- 🕸️ Selenium + BeautifulSoup-based robust web scraper

---

## 📁 Project Structure

```
Hotel_Chatbot_LLM/
│
├── data/
│   └── raw/ taj-mahal-palace-mumbai.json                 # Raw scraped content
│   └── processed/ processed_dataset.json,                # Cleaned text files per hotel
│                   falcon_finetune_dataset.jsonl         # Instructions Based jsonl file
│   └── embeddings/rag_db                                 # ChromaDB Vector Files
├── model/                                               # Fine Tuned Falcon Model
│   └── checkpoint-20/ 
│   └── checkpoint-20/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── README.md
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └──tokenizer.json
├── offload/                                              #Offload Directory
├── scripts/
│   ├── 01_scraper.py               # Web scraper using Selenium + fake-useragent + BeautifulSoup
│   ├── 02_preprocess.py            # Converts scraped data into instruction-tuned format
│   ├── 03_instruct_dataset.py      # Converts scraped data into instruction-tuned format
│   ├── 04_fine_tune.py             # LoRA fine-tuning script using HuggingFace Transformers
│   ├── 05_inference.py             # Script for testing model responses
│   └── 06_rag_pipeline.py          # ChromaDB RAG pipeline
├── app.py                          # Streamlit chat interface
├── requirements.txt                # Dependencies 
└── README.md

```

---

## 🧠 Model Details

- **Base Model**: `tiiuae/Falcon-1B-Instruct`
- **Fine-tuning Technique**:  QLoRA (using `peft`)
- **Tokenizer**: AutoTokenizer from HuggingFace
- **Training Format**:
  ```json
  {
    "text": "### Instruction:\n<QUESTION>\n\n### Response:\n<ANSWER>"
  }
  ```

---

## ⚒️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/hotel-chatbot-llm.git
cd hotel-chatbot-llm
```

### 2. Create & Activate Virtual Environment
```bash
conda create -p venv python==3.10 -y
conda activate venv/
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Scrape and Prepare Dataset
```bash
python scripts/01_scraper.py
python scripts/02_preprocess.py
python scripts/03_instruct.py
```

### 5. Fine-tune the Model
```bash
python scripts/04_fine_tune.py
```

### 6. Prepare VectorDB
```bash
python scripts/06_rag_pipeline.py
```

### 7. Launch Chatbot
```bash
streamlit run app.py
```
---

## 📊 Example Prompt Format

```
### Instruction:
What are the check-in and check-out timings at JW Marriott Bengaluru?

### Response:
Check-in time is from 3:00 PM, and check-out is until 12:00 PM.
```

---

## 💡 Future Enhancements

- Better Knowledge Base with multiple hotels for no RAG approach
- Faster Retrival 
- Experiments with other LLM models for better results without hallucinations
- Using API keys not local installation of LLM

---

## 🤝 Acknowledgements

- HuggingFace Transformers & PEFT
- Falcon-1B-Instruct from tiiuae
- Selenium + BeautifulSoup for scraping
- Streamlit for UI

---

