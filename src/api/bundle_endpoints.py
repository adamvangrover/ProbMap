from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

router = APIRouter(tags=["bundle"])

class CreditMemoRequest(BaseModel):
    text: str

class AnalysisResponse(BaseModel):
    risk_rating: str
    conviction_score: float
    semantic_breakdown: List[str]

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_credit_memo(request: CreditMemoRequest):
    """
    Analyzes a credit memo text using the Adam Sovereign Credit Bundle logic.
    """
    text = request.text
    if not text:
        raise HTTPException(status_code=400, detail="Text content is required")

    # 1. Conviction Score (Citation Density)
    sentences = text.count('.')
    citations = text.count('[doc_')

    conviction_score = 0.0
    if sentences > 0:
        conviction_score = citations / sentences

    # 2. Risk Rating (Keyword-based)
    risk_keywords = ["default", "bankruptcy", "insolvency", "restructuring", "loss", "decline"]
    risk_rating = "Low Risk"
    flagged_sentences = []

    # 3. Semantic Breakdown (Keyword extraction)
    for sentence in text.split('.'):
        sentence = sentence.strip()
        if not sentence:
            continue

        found_keywords = [kw for kw in risk_keywords if kw in sentence.lower()]
        if found_keywords:
            risk_rating = "High Risk"  # Simple logic: any keyword triggers High Risk
            flagged_sentences.append(f"Flagged: '{sentence}' (Keywords: {', '.join(found_keywords)})")

    return AnalysisResponse(
        risk_rating=risk_rating,
        conviction_score=conviction_score,
        semantic_breakdown=flagged_sentences
    )
