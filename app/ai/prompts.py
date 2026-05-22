"""
System and user prompt templates for the academic analysis LLM.
"""


def _build_output_schema(include_rubric: bool) -> str:
    """Build the JSON schema section expected from the model."""
    if include_rubric:
        return """{
  "jury_questions": [
    {
      "question": "The full question text",
      "rationale": "Why this question matters for the evaluation",
      "section_reference": "Which part of the document this targets"
    }
  ],
  "critical_remarks": [
    {
      "remark": "The specific criticism",
      "severity": "major | moderate | minor",
      "suggestion": "What the student should do to address this"
    }
  ],
  "improvement_suggestions": [
    {
      "area": "The area that needs improvement",
      "current_state": "What the document currently does",
      "recommended_action": "Specific steps to improve"
    }
  ],
  "scoring_rubric": {
    "methodology": {"score": 0, "max": 20, "comment": "..."},
    "clarity": {"score": 0, "max": 20, "comment": "..."},
    "technical_correctness": {"score": 0, "max": 20, "comment": "..."},
    "argumentation": {"score": 0, "max": 20, "comment": "..."},
    "originality": {"score": 0, "max": 20, "comment": "..."},
    "overall": {"score": 0, "max": 100, "comment": "..."}
  }
}"""

    return """{
  "jury_questions": [
    {
      "question": "The full question text",
      "rationale": "Why this question matters for the evaluation",
      "section_reference": "Which part of the document this targets"
    }
  ],
  "critical_remarks": [
    {
      "remark": "The specific criticism",
      "severity": "major | moderate | minor",
      "suggestion": "What the student should do to address this"
    }
  ],
  "improvement_suggestions": [
    {
      "area": "The area that needs improvement",
      "current_state": "What the document currently does",
      "recommended_action": "Specific steps to improve"
    }
  ]
}"""


def build_system_prompt(include_rubric: bool = True) -> str:
    """Build the strict instruction prompt for the analysis model."""
    schema = _build_output_schema(include_rubric=include_rubric)
    rubric_rule = "- Fill all scoring rubric fields.\n" if include_rubric else "- Do not include scoring_rubric in the response.\n"

    return f"""You are a **strict, experienced academic jury member** evaluating a student's academic document (thesis, dissertation, or presentation).

Your role:
- Critically evaluate the document's **methodology**, **clarity**, **technical correctness**, **argumentation**, and **overall academic rigor**.
- Be **precise**, **structured**, and **demanding** in your assessments.
- **Never give vague or generic feedback.** Every remark must be specific and tied to concrete observations from the document.
- Focus on what the student **should have done differently** or **must improve**.

You must respond **strictly** in the following JSON format (no markdown, no extra text):

{schema}

Rules:
- Generate **exactly 5** jury questions.
- Generate **exactly 5** critical remarks.
- Generate **exactly 3** improvement suggestions.
{rubric_rule}- Be specific — reference the exact content you're evaluating.
- **CRITICAL: YOUR ENTIRE RESPONSE (all keys and values) MUST BE IN FRENCH.**
"""


def build_analysis_prompt(
    context_chunks: list[str],
    document_title: str,
    reference_questions: list[str] | None = None,
    custom_query: str | None = None,
    include_rubric: bool = True,
) -> str:
    """Build the user prompt with RAG context and reference questions.

    Args:
        context_chunks: Retrieved relevant text chunks from the document.
        document_title: Name/title of the document being analyzed.
        reference_questions: List of relevant academic jury questions for inspiration.
        custom_query: Optional additional focus area from the user.
        include_rubric: Whether the response should include the scoring rubric.

    Returns:
        The formatted user prompt string.
    """
    context_block = "\n\n---\n\n".join(
        f"[Chunk {i + 1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
    )

    prompt = f"""## Document Under Review
**Title:** {document_title}

## Extracted Content (Most Relevant Sections)

{context_block}

## Your Task
Analyze the above document content and produce:
1. **5 jury-style questions** that a defense committee would ask — challenging, precise, and focused on weaknesses or areas needing clarification.
2. **5 critical remarks** about the document's methodology, argumentation, structure, or technical accuracy.
3. **3 improvement suggestions** with concrete, actionable steps.
"""

    if include_rubric:
        prompt += """
4. **A scoring rubric** evaluating methodology, clarity, technical correctness, argumentation, and originality (each out of 20, total out of 100).
"""

    if reference_questions:
        ref_block = "\n".join(f"- {q}" for q in reference_questions)
        prompt += f"""
## Reference Questions (For Inspiration)
The following are high-quality academic questions previously used in similar contexts. Use these to guide the tone, rigor, and depth of your own questions, adapting them specifically to the document content:

{ref_block}
"""

    if custom_query:
        prompt += f"""
## Additional Focus Area (User Request)
{custom_query}
Please give special attention to this area in your analysis.
"""

    if include_rubric:
        prompt += """
Respond ONLY with the JSON object as specified. No markdown formatting, no code fences, no extra text.
"""
    else:
        prompt += """
Respond ONLY with the JSON object as specified. Do not include a scoring rubric. No markdown formatting, no code fences, no extra text.
"""

    return prompt


SYSTEM_PROMPT = build_system_prompt(include_rubric=True)
