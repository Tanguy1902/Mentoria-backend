"""
Analysis service: RAG orchestration — retrieve, prompt, call LLM, parse results.
"""

from dataclasses import dataclass, field

from app.ai.llm_client import LLMClient
from app.ai.prompts import SYSTEM_PROMPT, build_analysis_prompt
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
        logger.info("Starting analysis for document: %s", document_id)

        # 1. Verify document exists
        doc = get_document(document_id)

        # 2. Verify document is indexed
        collection_name = f"doc_{document_id}"
        if not self._vector_store.collection_exists(collection_name):
            raise DocumentNotIndexedError(
                f"Document '{document_id}' has not been indexed yet",
                detail="Index the document first using POST /api/v1/index",
            )

        # 3. Retrieve relevant chunks
        search_query = custom_query or (
            "methodology, results, conclusion, technical approach, "
            "argumentation, literature review, experimental design"
        )

        retrieved = self._vector_store.query(
            collection_name=collection_name,
            query_text=search_query,
            n_results=settings.TOP_K_RESULTS,
        )

        if not retrieved:
            raise AnalysisError(
                "No relevant chunks found in the vector store",
                detail="The document may not have been properly indexed.",
            )

        context_chunks = [chunk.text for chunk in retrieved]
        logger.info("Retrieved %d chunks for analysis", len(context_chunks))

        # 3.b Retrieve reference questions for inspiration
        reference_questions = []
        if self._vector_store.collection_exists(REFERENCE_COLLECTION):
            # We use the same query to find questions in the same domain/topic
            ref_results = self._vector_store.query(
                collection_name=REFERENCE_COLLECTION,
                query_text=search_query,
                n_results=10,  # Get up to 10 relevant questions
            )
            reference_questions = [r.text for r in ref_results]
            logger.info("Retrieved %d reference questions for inspiration", len(reference_questions))
        else:
            logger.warning("Reference questions collection '%s' not found", REFERENCE_COLLECTION)

        # 4. Build prompt and call LLM
        user_prompt = build_analysis_prompt(
            context_chunks=context_chunks,
            document_title=doc.filename,
            reference_questions=reference_questions,
            custom_query=custom_query,
        )

        try:
            llm_response = await self._llm.generate_json(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
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
        return result

    async def chat_with_document(
        self,
        document_id: str,
        message: str,
        chat_history: list[dict] | None = None,
    ):
        """Interact with the document using the academic jury persona (streaming)."""
        settings = get_settings()
        doc = get_document(document_id)
        collection_name = f"doc_{document_id}"

        if not self._vector_store.collection_exists(collection_name):
            raise DocumentNotIndexedError(f"Document '{document_id}' not indexed")

        # 1. Retrieve context for the specific question
        retrieved = self._vector_store.query(
            collection_name=collection_name,
            query_text=message,
            n_results=5,
        )
        context = "\n\n".join([r.text for r in retrieved])

        # 2. Build chat prompt
        system_prompt = (
            "Vous êtes un membre du jury académique. Vous discutez avec l'étudiant "
            "de son travail basé sur les extraits fournis. Soyez exigeant, professionnel "
            "et précis. Répondez TOUJOURS en Français.\n\n"
            f"CONTEXTE DU DOCUMENT :\n{context}"
        )
        
        # 3. Stream from LLM
        async for chunk in self._llm.stream(
            system_prompt=system_prompt,
            user_prompt=message,
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
