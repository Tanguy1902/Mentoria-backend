import json
import os
import sys
import uuid
from pathlib import Path

# Add the project root to sys.path to allow imports from app
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.vectordb.chroma_client import ChromaVectorStore
from app.core.logging import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

REFERENCE_QUESTIONS_PATH = Path("app/data/questions_reference.json")
COLLECTION_NAME = "reference_questions"

def index_questions():
    """Index the reference questions into a dedicated ChromaDB collection."""
    if not REFERENCE_QUESTIONS_PATH.exists():
        logger.error(f"Reference questions file not found at {REFERENCE_QUESTIONS_PATH}")
        return

    logger.info(f"Reading questions from {REFERENCE_QUESTIONS_PATH}")
    with open(REFERENCE_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    logger.info(f"Found {len(questions)} questions. Initializing Vector Store...")
    
    vector_store = ChromaVectorStore()
    vector_store.initialize()

    # Prepare data for batch insert
    ids = []
    documents = []
    metadatas = []

    for q in questions:
        # Generate a unique ID for each question
        q_id = f"ref_{uuid.uuid4().hex[:12]}"
        ids.append(q_id)
        
        # The document is the question text itself
        documents.append(q["question"])
        
        # Metadata for filtering and context
        metadatas.append({
            "type": q.get("type", "general"),
            "role": q.get("role", "examinateur"),
            "niveau": q.get("niveau", "master"),
            "domaine": q.get("domaine", "general"),
            "difficulte": q.get("difficulte", 3),
            "frequence": q.get("frequence", 0.5),
            "tags": ",".join(q.get("tags", []))
        })

    logger.info(f"Indexing {len(documents)} questions into collection '{COLLECTION_NAME}'...")
    
    # Check if collection already exists and clear it if necessary (optional)
    if vector_store.collection_exists(COLLECTION_NAME):
        logger.info(f"Collection '{COLLECTION_NAME}' already exists. Deleting it for a clean re-index...")
        vector_store.delete_collection(COLLECTION_NAME)

    count = vector_store.add_chunks(
        collection_name=COLLECTION_NAME,
        chunk_ids=ids,
        documents=documents,
        metadatas=metadatas
    )

    logger.info(f"Successfully indexed {count} reference questions.")

if __name__ == "__main__":
    index_questions()
