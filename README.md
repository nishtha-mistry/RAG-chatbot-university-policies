# 🤖 AI-Powered RAG Chatbot for University Policies

## 🔍 Overview

This project implements a **Retrieval-Augmented Generation (RAG)** based chatbot designed to answer student queries using university policy documents.

The system processes multiple documents, performs semantic search using vector embeddings, and generates accurate answers using an LLM.

It enables students to quickly access policy-related information without manually searching through lengthy documents.

---

## 🚀 Features

* 📄 Supports PDF and DOCX document ingestion
* 🔍 Semantic search using FAISS vector database
* 🧠 Context-aware answer generation using FLAN-T5
* ✂️ Smart text chunking with overlap
* ⚡ Fast retrieval using dense embeddings
* 🌐 Interactive UI using Streamlit
* 💾 Save & load FAISS index for reuse

---

## 🧠 How It Works (RAG Pipeline)

1. **Document Ingestion**

   * Extracts text from PDF/DOCX using `pdfplumber` and `python-docx`

2. **Text Preprocessing**

   * Cleans and structures extracted text
   * Converts tables into readable sentences

3. **Chunking**

   * Splits text into overlapping chunks for better context retention

4. **Embedding**

   * Uses Sentence Transformers (`all-mpnet-base-v2`) to generate vector embeddings

5. **Vector Storage**

   * Stores embeddings in FAISS index for efficient similarity search

6. **Retrieval**

   * Retrieves top-k relevant chunks based on user query

7. **Answer Generation**

   * Uses FLAN-T5 to generate concise answers from retrieved context

---

## 🛠️ Tech Stack

* Python 🐍
* FAISS
* Sentence Transformers
* Hugging Face Transformers (FLAN-T5)
* Streamlit
* pdfplumber
* python-docx
* NumPy / Pandas

---

## 📂 Project Structure

```id="u7k29a"
📁 rag-chatbot-university-policies
│
├── 📄 rag-chatbot-university-policies.py
├── 📄 requirements.txt
├── 📄 README.md
```

---

## ▶️ How to Run (Important ⚠️)

Since this project was built in a local environment (VS Code), follow these steps carefully:

### 1. Clone Repository

```bash id="qk12p0"
git clone https://github.com/your-username/rag-chatbot-university-policies.git
cd rag-chatbot-university-policies
```

### 2. Create Virtual Environment

#### Windows:

```bash id="n91d2a"
python -m venv venv
venv\Scripts\activate
```

#### Mac/Linux:

```bash id="f0x2l1"
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash id="v8p3m1"
pip install -r requirements.txt
```

---

### 4. Run the Application

```bash id="l2x9s0"
streamlit run rag-chatbot-university-policies.py
```

---

## ⚙️ Usage

1. Upload university policy documents (PDF/DOCX)
2. Configure chunk size, overlap, and embedding settings
3. Click **"Build Index"**
4. Ask questions in natural language
5. View:

   * Retrieved document chunks
   * Generated answer (FLAN-T5)

---

## 📊 Key Highlights

* Built a **complete RAG pipeline from scratch**
* Implemented **semantic chunking and retrieval**
* Enabled **context-aware question answering**
* Designed a **user-friendly Streamlit interface**
* Optimized retrieval using FAISS similarity search

---

## ⚠️ Challenges Faced

* Environment setup and dependency management in VS Code
* Handling large document processing efficiently
* Balancing chunk size vs retrieval accuracy
* Managing model latency during answer generation

---

## 🔮 Future Improvements

* Add GPT-based API for higher-quality responses
* Deploy on cloud (AWS / Hugging Face Spaces)
* Add authentication for student-specific queries
* Improve UI/UX for better interaction
* Add conversation memory

---

## 📬 Contact

Feel free to connect for questions, improvements, or collaborations!

---

⭐ If you found this project useful, consider starring the repo!
