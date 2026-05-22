"""
Analysis service: RAG orchestration — retrieve, prompt, call LLM, parse results.
"""

import time
from dataclasses import dataclass, field

import anyio

from app.ai.llm_client import LLMClient
from app.ai.prompts import build_analysis_prompt, build_system_prompt
from app.config import get_settings
from app.core.exceptions import AnalysisError, DocumentNotIndexedError
from app.core.logging import get_logger
from app.services.document_service import get_document
from app.vectordb.chroma_client import ChromaVectorStore

logger = get_logger(__name__)

REFERENCE_COLLECTION = "reference_questions"


@dataclass
class JuryQuestion:
    question: str
    rationale: str
    section_reference: str


@dataclass
class CriticalRemark:
    remark: str
    severity: str
    suggestion: str


@dataclass
class ImprovementSuggestion:
    area: str
    current_state: str
    recommended_action: str


@dataclass
class RubricScore:
    score: int
    max: int
    comment: str


@dataclass
class ScoringRubric:
    methodology: RubricScore
    clarity: RubricScore
    technical_correctness: RubricScore
    argumentation: RubricScore
    originality: RubricScore
    overall: RubricScore


@dataclass
class AnalysisResult:
    """Full structured analysis output from the LLM."""
    document_id: str
    document_title: str
    jury_questions: list[JuryQuestion] = field(default_factory=list)
    critical_remarks: list[CriticalRemark] = field(default_factory=list)
    improvement_suggestions: list[ImprovementSuggestion] = field(default_factory=list)
    scoring_rubric: ScoringRubric | None = None
    model_used: str = ""
    chunks_used: int = 0


def _limit_texts(texts: list[str], char_limit: int) -> list[str]:
    """Trim a list of texts so the combined size stays under a character budget."""
    limited: list[str] = []
    current_size = 0

    for text in texts:
        if current_size >= char_limit:
            break

        remaining = char_limit - current_size
        if len(text) > remaining:
            trimmed = text[:remaining].rstrip()
            if trimmed:
                limited.append(f"{trimmed}...")
            break

        limited.append(text)
        current_size += len(text)

    return limited


def _format_chat_history(chat_history: list[dict] | None, max_turns: int) -> str:
    """Format a trimmed chat history for inclusion in the prompt."""
    if not chat_history:
        return "Aucun échange précédent."

    recent_messages = chat_history[-(max_turns * 2):]
    lines: list[str] = []
    for message in recent_messages:
        role = message.get("role", "user")
        content = str(message.get("content", "")).strip()
        if not content:
            continue

        if len(content) > 500:
            content = content[:500].rstrip() + "..."

        speaker = "Utilisateur" if role == "user" else "Jury"
        lines.append(f"{speaker}: {content}")

    return "\n".join(lines) if lines else "Aucun échange précédent."


class AnalysisService:
    """Orchestrates the RAG analysis pipeline."""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        llm_client: LLMClient,
    ) -> None:
        self._vector_store = vector_store
        self._llm = llm_client

    async def analyze_document(
        self,
        document_id: str,
        custom_query: str | None = None,
        include_rubric: bool = True,
    ) -> AnalysisResult:
        """Run the full RAG analysis pipeline for a document.

        Steps:
        1. Verify document exists and is indexed
        2. Retrieve most relevant chunks from vector store
        3. Build prompt with context
        4. Call LLM for structured analysis
        5. Parse and return structured result

        Args:
            document_id: The document to analyze.
            custom_query: Optional specific focus area.
            include_rubric: Whether to include scoring rubric.

        Returns:
            A fully structured ``AnalysisResult``.

        Raises:
            DocumentNotIndexedError: If the document hasn't been indexed yet.
            AnalysisError: If the analysis pipeline fails.
        """
        settings = get_settings()
        started_at = time.perf_counter()
        logger.info("Starting analysis for document: %s", document_id)

        # 1. Verify document exists
        doc = await anyio.to_thread.run_sync(get_document, document_id)

        # 2. Verify document is indexed
        collection_name = f"doc_{document_id}"
        collection_exists = await anyio.to_thread.run_sync(
            self._vector_store.collection_exists,
            collection_name,
        )
        if not collection_exists:
            raise DocumentNotIndexedError(
                f"Document '{document_id}' has not been indexed yet",
                detail="Index the document first using POST /api/v1/index",
            )

        # 3. Retrieve relevant chunks
        search_query = custom_query or (
            "methodology, results, conclusion, technical approach, "
            "argumentation, literature review, experimental design"
        )

        retrieved = await anyio.to_thread.run_sync(
            self._vector_store.query,
            collection_name,
            search_query,
            settings.TOP_K_RESULTS,
        )

        if not retrieved:
            raise AnalysisError(
                "No relevant chunks found in the vector store",
                detail="The document may not have been properly indexed.",
            )

        context_chunks = _limit_texts(
            [chunk.text for chunk in retrieved],
            settings.ANALYSIS_CONTEXT_CHAR_LIMIT,
        )
        logger.info("Retrieved %d chunks for analysis", len(context_chunks))

        # 3.b Retrieve reference questions for inspiration
        reference_questions = []
        reference_collection_exists = await anyio.to_thread.run_sync(
            self._vector_store.collection_exists,
            REFERENCE_COLLECTION,
        )
        if reference_collection_exists:
            # We use the same query to find questions in the same domain/topic
            ref_results = await anyio.to_thread.run_sync(
                self._vector_store.query,
                REFERENCE_COLLECTION,
                search_query,
                settings.ANALYSIS_REFERENCE_LIMIT,
            )
            reference_questions = _limit_texts(
                [r.text for r in ref_results],
                settings.ANALYSIS_CONTEXT_CHAR_LIMIT // 3,
            )
            logger.info("Retrieved %d reference questions for inspiration", len(reference_questions))
        else:
            logger.warning("Reference questions collection '%s' not found", REFERENCE_COLLECTION)

        # 4. Build prompt and call LLM
        user_prompt = build_analysis_prompt(
            context_chunks=context_chunks,
            document_title=doc.filename,
            reference_questions=reference_questions,
            custom_query=custom_query,
            include_rubric=include_rubric,
        )
        system_prompt = build_system_prompt(include_rubric=include_rubric)
        max_tokens = settings.LLM_MAX_TOKENS if include_rubric else min(settings.LLM_MAX_TOKENS, 2500)

        try:
            llm_response = await self._llm.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            logger.error("LLM analysis failed: %s", exc)
            raise AnalysisError(
                "Failed to generate analysis from LLM",
                detail=str(exc),
            ) from exc

        # 5. Parse structured response
        try:
            result = self._parse_llm_response(
                llm_response,
                document_id=document_id,
                document_title=doc.filename,
                chunks_used=len(context_chunks),
                include_rubric=include_rubric,
            )
        except Exception as exc:
            logger.error("Failed to parse LLM response: %s", exc)
            raise AnalysisError(
                "Failed to parse structured analysis from LLM response",
                detail=str(exc),
            ) from exc

        logger.info("Analysis complete for document: %s", document_id)
        logger.info("Analysis pipeline finished in %.2fs", time.perf_counter() - started_at)
        return result

    async def chat_with_document(
        self,
        document_id: str,
        message: str,
        chat_history: list[dict] | None = None,
    ):
        """Interact with the document using the academic jury persona (streaming)."""
        settings = get_settings()
        doc = await anyio.to_thread.run_sync(get_document, document_id)
        collection_name = f"doc_{document_id}"

        collection_exists = await anyio.to_thread.run_sync(
            self._vector_store.collection_exists,
            collection_name,
        )
        if not collection_exists:
            raise DocumentNotIndexedError(f"Document '{document_id}' not indexed")

        # 1. Retrieve context for the specific question
        retrieved = await anyio.to_thread.run_sync(
            self._vector_store.query,
            collection_name,
            message,
            min(3, settings.TOP_K_RESULTS),
        )
        context = "\n\n".join(
            _limit_texts([r.text for r in retrieved], settings.CHAT_CONTEXT_CHAR_LIMIT)
        )
        history_block = _format_chat_history(chat_history, settings.CHAT_HISTORY_TURNS)

        # 2. Build chat prompt
        system_prompt = (
            "Vous êtes un membre du jury académique. Vous discutez avec l'étudiant "
            "de son travail basé sur les extraits fournis. Soyez exigeant, professionnel "
            "et précis. Répondez TOUJOURS en Français.\n\n"
            f"DOCUMENT : {doc.filename}\n\n"
            f"CONTEXTE DU DOCUMENT :\n{context}\n\n"
            f"HISTORIQUE RÉCENT :\n{history_block}"
        )
        
        # 3. Stream from LLM
        async for chunk in self._llm.stream(
            system_prompt=system_prompt,
            user_prompt=message,
            max_tokens=min(settings.LLM_MAX_TOKENS, 1500),
        ):
            yield chunk

    def _parse_llm_response(
        self,
        data: dict,
        document_id: str,
        document_title: str,
        chunks_used: int,
        include_rubric: bool,
    ) -> AnalysisResult:
        """Parse the raw LLM JSON into typed dataclasses."""

        # Parse jury questions
        questions: list[JuryQuestion] = []
        for q in data.get("jury_questions", []):
            questions.append(
                JuryQuestion(
                    question=q.get("question", ""),
                    rationale=q.get("rationale", ""),
                    section_reference=q.get("section_reference", ""),
                )
            )

        # Parse critical remarks
        remarks: list[CriticalRemark] = []
        for r in data.get("critical_remarks", []):
            remarks.append(
                CriticalRemark(
                    remark=r.get("remark", ""),
                    severity=r.get("severity", "moderate"),
                    suggestion=r.get("suggestion", ""),
                )
            )

        # Parse improvement suggestions
        suggestions: list[ImprovementSuggestion] = []
        for s in data.get("improvement_suggestions", []):
            suggestions.append(
                ImprovementSuggestion(
                    area=s.get("area", ""),
                    current_state=s.get("current_state", ""),
                    recommended_action=s.get("recommended_action", ""),
                )
            )

        # Parse scoring rubric
        rubric: ScoringRubric | None = None
        if include_rubric and "scoring_rubric" in data:
            rb = data["scoring_rubric"]
            rubric = ScoringRubric(
                methodology=RubricScore(**rb.get("methodology", {"score": 0, "max": 20, "comment": "N/A"})),
                clarity=RubricScore(**rb.get("clarity", {"score": 0, "max": 20, "comment": "N/A"})),
                technical_correctness=RubricScore(**rb.get("technical_correctness", {"score": 0, "max": 20, "comment": "N/A"})),
                argumentation=RubricScore(**rb.get("argumentation", {"score": 0, "max": 20, "comment": "N/A"})),
                originality=RubricScore(**rb.get("originality", {"score": 0, "max": 20, "comment": "N/A"})),
                overall=RubricScore(**rb.get("overall", {"score": 0, "max": 100, "comment": "N/A"})),
            )

        return AnalysisResult(
            document_id=document_id,
            document_title=document_title,
            jury_questions=questions,
            critical_remarks=remarks,
            improvement_suggestions=suggestions,
            scoring_rubric=rubric,
            model_used=get_settings().OPENROUTER_MODEL,
            chunks_used=chunks_used,
        )
