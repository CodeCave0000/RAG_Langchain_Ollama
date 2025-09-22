# RAG_Langchain_Ollama

This project demonstrates how to build a **retrieval-augmented question-answering (QA) pipeline** over a PDF document using [LangChain](https://www.langchain.com/), [FAISS](https://faiss.ai/), and [Ollama](https://ollama.ai/).

The script loads a PDF receipt, splits it into chunks, generates vector embeddings, and retrieves the most relevant context to answer a query using a local LLM.

🎥 **Related YouTube Video**: [Watch here](https://youtu.be/7oQg8kVVXRg?si=fwJd-Ph7hh-qd6oe)

---

## 📌 Features
- **PDF Loading**: Extracts text from a PDF file using `PyPDFLoader`.
- **Chunking**: Splits large documents into overlapping text chunks for efficient embedding.
- **Embeddings**: Uses `OllamaEmbeddings` (`nomic-embed-text` model) to create vector representations.
- **Vector Search**: Stores and retrieves document chunks with `FAISS`.
- **Custom Prompting**: Uses `ChatPromptTemplate` with a configurable system role.
- **LLM Response**: Generates answers with `ChatOllama` (`llama3.1:8b` model).

---

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/CodeCave0000/RAG_Langchain_Ollama.git
cd RAG_Langchain_Ollama
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3. Install dependencies
```bash
pip install langchain langchain-community langchain-ollama langchain-text-splitters 
faiss-cpu pypdf
```

⚠️ Ensure you have [Ollama](https://ollama.ai/) installed and running locally.

---

## 🚀 Usage

Update the file path in the script with the location of your PDF:

```python
loader = PyPDFLoader("C:\\Users\\ranya\\OneDrive\\Desktop\\SamsungGalaxyTabS3.pdf")
```

Run the script:
```bash
python main.py
```

Example query inside the script:
```python
query = "What was purchased in this receipt?"
```

Expected output (LLM response with cynicism applied 😉):
```
Of course, you spent money on yet another gadget... The Samsung Galaxy Tab S3.
```
---

## 🔧 Customization
- **Change the system role** in `ChatPromptTemplate` to adjust tone/behavior:
  ```python
  ("system", "You are a helpful financial assistant")
  ```
- **Adjust chunk size** in `RecursiveCharacterTextSplitter`:
  ```python
  splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
  ```
- **Swap Ollama model**:
  ```python
  llm = ChatOllama(model="llama2:7b")
  ```

---

## 📌 Requirements
- Python 3.9+
- Ollama installed and running locally
- LangChain + FAISS

---

## ⚖️ License
MIT License. Feel free to use and adapt.
