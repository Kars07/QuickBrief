# **QuickBrief**   
**AI-Powered PDF Summarization & Q&A Tool**  

QuickBrief is an intelligent agent that ingests PDFs, extracts key information, generates summaries, and answers questions about the document content. Built with LangChain and NVIDIA AI models, it provides fast, accurate insights from your documents.  

---

## **Features**  
- **Document Ingestion**: Loads PDFs and text files from a directory.  
- **Smart Summarization**: Generates concise yet detailed summaries.  
- **Q&A Capabilities**: Answers questions based on document content.  
- **Vector Search**: Uses FAISS for efficient document retrieval.  
- **Auto-Summarization**: Optionally summarizes documents on first interaction.  

---

## **Tech Stack & Models**  
### **Core Frameworks**  
- **LangChain** (Document processing, retrieval, agent orchestration)  
- **FAISS** (Vector similarity search)  
- **Pydantic** (Configuration & data validation)  

### **Embedding Model**  
- **`nvidia/nv-embedqa-e5-v5`** (NVIDIA Embeddings for semantic search)  

### **LLM for Summarization & Q&A**  
- **`meta/llama-4-maverick-17b-128e-instruct`** (via NVIDIA AI Endpoints)  

---

## **⚙️ Setup & Installation**  

### **Prerequisites**  
- Python 3.10+  
- NVIDIA API Key (for `ChatNVIDIA`)  
- Required packages (`langchain`, `faiss-cpu`, `pypdf`, etc.)  

### **Installation**  
1. Clone the repository:  
   ```bash
   git clone https://github.com/your-repo/QuickBrief.git
   cd QuickBrief
   ```

2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables (`.env`):  
   ```plaintext
   NVIDIA_API_KEY=your_api_key_here
   ```

4. Run the tool:  
   ```bash
   aiq serve --config_file note_summarization/configs/config.yml --host 0.0.0.0 --port 8000
   ```

---

## **Configuration**  
The tool is configured via `NoteSummarizationFunctionConfig`:  

| Parameter         | Description | Default |
|-------------------|-------------|---------|
| `ingest_glob`     | File pattern for PDF/TXT ingestion (e.g., `./docs/*.pdf`) | Required |
| `llm_name`        | LLM reference (`meta/llama-4-maverick-17b-128e-instruct`) | Required |
| `chunk_size`      | Text split size for processing | `1024` |
| `auto_summarize`  | Auto-summarize on first message | `True` |
| `embedder_name`   | Embedding model (`nvidia/nv-embedqa-e5-v5`) | `"nvidia/nv-embedqa-e5-v5"` |

---

## **Usage**  
1. **Place documents** in the specified directory (e.g., `./docs/`).  
2. **Start the agent**—it will auto-summarize documents if enabled.  
3. **Ask questions** like:  
   - *"Summarize the key points of this document."*  
   - *"What does section 3 discuss?"*  
   - *"List the main recommendations."*  

---

## **Troubleshooting**  
- **Error loading files?** Check file paths and permissions.  
- **No documents found?** Verify `ingest_glob` points to the right directory.  
- **LLM not responding?** Ensure the NVIDIA API key is valid.  

---

## **License**  
MIT License  

---

**QuickBrief** – Turn documents into insights in seconds! 🚀  
*Built with LangChain & NVIDIA AI.*
