import json
import uuid
from pathlib import Path
from typing import List
from fastapi import APIRouter, HTTPException, Depends
from app.api.schemas.reference import ReferenceQuestionContribute
from app.api.dependencies import get_vector_store
from app.vectordb.chroma_client import ChromaVectorStore
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

CONTRIBUTIONS_PATH = Path("app/data/contributions_real.json")


@router.post(
    "/contribute",
    summary="Contribute a reference question",
    description="Save a real academic question to both the contribution database and ChromaDB.",
)
async def contribute_question(
    question: ReferenceQuestionContribute,
    vector_store: ChromaVectorStore = Depends(get_vector_store),
):
    try:
        CONTRIBUTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)

        data = []
        if CONTRIBUTIONS_PATH.exists():
            with open(CONTRIBUTIONS_PATH, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    logger.warning("Contributions file was corrupted, starting fresh.")
                    data = []

        data.append(question.dict())

        with open(CONTRIBUTIONS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        q_id = f"ref_{uuid.uuid4().hex[:12]}"
        metadata = {
            "type": question.type,
            "role": question.role,
            "niveau": question.niveau,
            "domaine": question.domaine,
            "difficulte": question.difficulte,
            "frequence": question.frequence,
            "tags": ",".join(question.tags)
        }
        vector_store.add_chunks(
            collection_name="reference_questions",
            chunk_ids=[q_id],
            documents=[question.question],
            metadatas=[metadata]
        )

        logger.info("New reference question contributed and indexed successfully.")
        return {"status": "success", "message": "Question enregistrée et indexée avec succès"}

    except Exception as e:
        logger.error(f"Failed to save contribution: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Erreur lors de la sauvegarde de la question",
        )


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
            data.append(q.dict())

        with open(CONTRIBUTIONS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        ids = []
        documents = []
        metadatas = []
        for q in questions:
            q_id = f"ref_{uuid.uuid4().hex[:12]}"
            ids.append(q_id)
            documents.append(q.question)
            metadatas.append({
                "type": q.type,
                "role": q.role,
                "niveau": q.niveau,
                "domaine": q.domaine,
                "difficulte": q.difficulte,
                "frequence": q.frequence,
                "tags": ",".join(q.tags)
            })

        vector_store.add_chunks(
            collection_name="reference_questions",
            chunk_ids=ids,
            documents=documents,
            metadatas=metadatas
        )

        count = len(questions)
        logger.info(f"{count} reference questions contributed and indexed successfully.")
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
