from pydantic import BaseModel, Field
from typing import List

class ReferenceQuestionContribute(BaseModel):
    question: str = Field(..., description="Le texte de la question")
    type: str = Field(..., description="Le type de question (ex: méthodologie, résultats, etc.)")
    role: str = Field(..., description="Le rôle attendu (ex: examinateur, rapporteur)")
    niveau: str = Field(..., description="Le niveau académique (ex: master, licence, doctorat)")
    domaine: str = Field(..., description="Le domaine d'étude (ex: informatique, économie)")
    tags: List[str] = Field(default_factory=list, description="Liste de mots-clés")
    frequence: float = Field(0.5, ge=0, le=1, description="Fréquence d'apparition (0-1)")
    difficulte: int = Field(3, ge=1, le=5, description="Niveau de difficulté (1-5)")
