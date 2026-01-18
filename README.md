# Haqdarshak Scheme Research Tool (SRT) 🔍

A Retrieval-Augmented Generation (RAG) application designed to automate research on government schemes. This tool ingests scheme-related URLs (articles or PDFs), generates structured summaries (Benefits, Application Process, Eligibility, Documents), and allows users to ask specific questions about the schemes using an interactive chat interface.

## 🚀 Features

* **Modular Ingestion Pipeline:** Robust content extraction capable of handling raw HTML and PDFs.
* **Intelligent Fallback Mechanisms:** Implements a multi-stage loading strategy (UnstructuredURL -> Request Download -> UnstructuredFile -> PyPDF/BeautifulSoup) to maximize data retrieval success.
* **Automated Summarization:** auto-generates structured summaries covering Scheme Benefits, Application Process, Eligibility, and Required Documents.
* **Context-Aware Q&A:** Uses OpenAI's GPT models to answer user queries based *only* on the ingested source material to minimize hallucinations.
* **Persistent Vector Storage:** Saves FAISS indexes to a local pickle file (`faiss_store_openai.pkl`) for efficient reuse.

## 📂 Folder Structure

```text
.
├── .config                  # Environment variables (API Keys)
├── .gitignore
├── Dockerfile               # Container configuration
├── faiss_store_openai.pkl   # Serialized vector database (generated)
├── main.py                  # Streamlit application entry point
└── requirements.txt         # Python dependencies

```

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **LLM Integration:** LangChain, OpenAI (GPT-4.1 / GPT-3.5)
* **Vector Database:** FAISS
* **Document Loading:** Unstructured, PyPDF, BeautifulSoup4

## 🐳 Docker Setup & Development Workflow

This project was developed entirely within a Docker container to ensure cross-platform consistency and dependency isolation.

### 1. Build the Image

Build the Docker image using the provided `Dockerfile`.

```bash
docker build -t hakdarshak-srt .

```

### 2. Run the Container

We mount the current directory (`%cd%` on Windows or `$(pwd)` on Linux/Mac) to `/app` inside the container. This allows for real-time code editing on your host machine while the app runs in the isolated container environment.

**Windows (Command Prompt):**

```cmd
docker run -it -p 8501:8501 -v "%cd%":/app hakdarshak-srt

```

**Linux / Mac / PowerShell:**

```bash
docker run -it -p 8501:8501 -v "$(pwd)":/app hakdarshak-srt

```

Once running, access the application at `http://localhost:8501`.

## ⚙️ Configuration

1. Create a `.config` file in the root directory.
2. Add your OpenAI API key:
```env
OPENAI_API_KEY=sk-your-api-key-here

```



## 🧠 Technical Implementation Details

### The Dual-Function Loading Strategy

During development, standard loaders often failed on varied web sources. To solve this, a custom two-function approach was implemented in `main.py`:

1. **`try_unstructured_url_loader`:** Attempts to load the URL directly using LangChain's Unstructured integration.
2. **`try_download_and_load`:** If the first method fails, the script manually downloads the content to a temporary file. It then attempts to parse it using `UnstructuredFileLoader`, falling back to `PyPDFLoader` (for PDFs) or `BeautifulSoup` (for raw HTML) if necessary.

This ensures high resilience against broken headers, file type mismatches, or download restrictions.

## ⚠️ Limitations & Scope for Improvement

* **Dynamic Content (JavaScript):** The current ingestion pipeline relies on static HTML parsing. It cannot scrape content from websites that rely heavily on client-side JavaScript rendering (SPA).
* *Improvement:* Integrate tools like Playwright or Selenium to handle dynamic content.


* **Duplicate URL Processing:** The system currently processes every URL entered, even if it has been processed before. This leads to redundant embedding costs and bloated vector stores.
* *Improvement:* Implement a hashing mechanism or a database lookup to check if a URL has already been indexed before processing.


* **Vector Store Concurrency:** The use of a simple `pickle` file for the vector store is not thread-safe for multiple simultaneous users.
* *Improvement:* Migrate to a dedicated vector database service like Pinecone or Weaviate for production scaling.
