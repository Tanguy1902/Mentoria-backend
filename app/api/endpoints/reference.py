import json
import time
import uuid
from pathlib import Path
from typing import List

import anyio
from fastapi import APIRouter, HTTPException, Depends
from app.api.schemas.reference import ReferenceQuestionContribute
from app.api.dependencies import get_vector_store
from app.vectordb.chroma_client import ChromaVectorStore
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

CONTRIBUTIONS_PATH = Path(__file__).resolve().parents[2] / "data" / "contributions_real.json"


def format_metadata(question: ReferenceQuestionContribute) -> dict:
    """Create a Chroma-friendly metadata payload for a reference question."""
    return {
        "type": question.type,
        "role": question.role,
        "niveau": question.niveau,
        "domaine": question.domaine,
        "tags": ",".join(question.tags),
        "frequence": question.frequence,
        "difficulte": question.difficulte,
    }


def _persist_contributions_file(questions: list[ReferenceQuestionContribute]) -> None:
    """Append the contributed questions to the local JSON archive."""
    CONTRIBUTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)

    data = []
    if CONTRIBUTIONS_PATH.exists():
        with open(CONTRIBUTIONS_PATH, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                logger.warning("Contributions file was corrupted, starting fresh.")
                data = []

    for q in questions:
        data.append(q.model_dump())

    with open(CONTRIBUTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


@router.post(
    "/contribute",
    summary="Contribute a reference question",
    description="Save a real academic question to both the contribution database and ChromaDB.",
)
async def contribute_question(
    question: ReferenceQuestionContribute,
    vector_store: ChromaVectorStore = Depends(get_vector_store),
):
    # Reuse the batch logic for a single contribution
    return await contribute_questions_batch(questions=[question], vector_store=vector_store)


@router.post(
    "/contribute/batch",
    summary="Contribute multiple reference questions at once",
    description="Save a batch of real academic questions to both the contribution database and ChromaDB.",
)
async def contribute_questions_batch(
    questions: List[ReferenceQuestionContribute],
    vector_store: ChromaVectorStore = Depends(get_vector_store),
):
    if not questions:
        raise HTTPException(status_code=400, detail="La liste de questions est vide.")

    try:
        started_at = time.perf_counter()
        await anyio.to_thread.run_sync(_persist_contributions_file, questions)

        ids = []
        documents = []
        metadatas = []
        for q in questions:
            q_id = f"ref_{uuid.uuid4().hex[:12]}"
            ids.append(q_id)
            documents.append(q.question)
            metadatas.append(format_metadata(q))

        indexed_count = await anyio.to_thread.run_sync(
            vector_store.add_chunks,
            "reference_questions",
            ids,
            documents,
            metadatas,
        )

        count = indexed_count
        logger.info(f"{indexed_count} reference questions contributed and indexed successfully.")
        logger.info("Reference contribution pipeline finished in %.2fs", time.perf_counter() - started_at)
        return {
            "status": "success",
            "message": f"{count} question(s) enregistrée(s) et indexée(s) avec succès",
            "count": count,
        }

    except Exception as e:
        logger.error(f"Failed to save batch contribution: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Erreur lors de la sauvegarde des questions",
        )
