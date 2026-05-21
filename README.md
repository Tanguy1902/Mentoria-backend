# AI Academic Document Analyzer

A robust, production-ready FastAPI backend designed to act as an academic jury. 
This system ingests student academic documents (PDF & PowerPoint presentations), processes them using a Retrieval-Augmented Generation (RAG) architecture, and utilizes OpenRouter LLMs to generate highly specific, structured feedback.

## 🧱 Architecture Overview

The software enforces a clean, modular layer architecture:

1. **API Layer (`app/api`):** Handles routing, data validation via Pydantic schemas, dependency injection, and endpoint controllers.
2. **Service Layer (`app/services`):** Heart of business logic orchestrating PDF/PPT extraction, context chunking, and final RAG pipeline assembly. 
3. **AI Layer (`app/ai`):** Prompts management, interactions with the local Embedding model (`sentence-transformers` via ChromaDB built-in features), and RESTful client calls for OpenRouter unified LLM endpoint processing.
4. **Vector Database Layer (`app/vectordb`):** Wrapping functions for `ChromaDB`, managing persistent document collections, embedding storage, and similarity search.

```text
Upload PDF/PPT → Text Extraction → Vector Chunking (tiktoken) → ChromaDB Persistence
                                                                          ↓
User Query → Similarity Search (Cosine) → Prompt Augmentation → OpenRouter LLM Generate
                                                                          ↓
                                            Jury Feedback (JSON Array Response)
```

## ⚙️ Core Stack Features
* **Web Framework:** FastAPI + Uvicorn + Pydantic
* **LLM Engine:** OpenRouter (OpenAI SDK compatibility layer)
* **Vector DB:** ChromaDB (local persistence, local default embedding model)
* **Parsers:** `PyMuPDF` for PDF, `python-pptx` for PowerPoint.
* **Text Processing:** `tiktoken` (tokenizer matching GPT styles), recursive overlap text chunking.

---

## 🚀 Environment Setup

### 1. Requirements

Ensure you are using Python 3.11+. Setup your environment:

```bash
# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Copy the example environment variables file and update your keys.

```bash
cp .env.example .env
```

Open `.env` and configure:
* `OPENROUTER_API_KEY`: Your OpenRouter api key. 
* `OPENROUTER_MODEL`: Target model (defaults to `google/gemini-2.0-flash-001` or another LLM of choice).

---

## ⚡ Running the API locally

Start the standard Uvicorn server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The server will automatically boot the ChromaDB persistence engine locally.
Swagger interface is accessible at: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📦 API Usage Examples

### 1. Health Verification
```bash
curl -X GET http://localhost:8000/api/v1/health
```

### 2. Upload a Document
Upload a PDF/PPT file. The system will save and extract its text internally.

```bash
curl -X POST http://localhost:8000/api/v1/upload \
  -F "file=@/path/to/my_presentation.pptx"
```
*Returns:* The extracted `document_id` used in subsequent steps.

### 3. Build the Vectors (Index)
Chunks the extracted text layout and stores vector embeddings in the DB.

```bash
curl -X POST http://localhost:8000/api/v1/index \
  -H "Content-Type: application/json" \
  -d '{"document_id": "<document_id_from_step_2>"}'
```

### 4. Perform the AI Analysis
Executes the RAG pipeline to generate the structured JSON Array containing jury-level questions, critical remarks, improvements, and an optional scoring rubric.

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"document_id": "<document_id_from_step_2>"}'
```

> **Note:** The `analyze` endpoint includes optional properties to filter the custom query explicitly (`custom_query`) or disable the scoring array entirely (`include_rubric: false`).
