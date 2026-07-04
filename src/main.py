from fastapi import FastAPI
from pydantic import BaseModel, Field
import pysbd
import numpy as np

from .providers import get_provider

app = FastAPI()

@app.get("/health")
def health() -> dict[str, str]:
    """Report service liveness."""
    return {"status": "ok"}


class MatchRequest(BaseModel):
    cv_bullets: list[str]
    jd_text: str
    jd_language: str = "en"
    threshold: float = Field(0.7, ge=0, le=1)
    top_n: int = Field(3, ge=1)

class BulletScore(BaseModel):
    bullet: str
    score: float

class RequirementMatch(BaseModel):
    requirement: str
    top_matches: list[BulletScore]

class MatchResponse(BaseModel):
    requirement_sentences: list[str]
    cv_bullets: list[str]
    matrix: list[list[float]]
    coverage_gaps: list[str]
    requirements: list[RequirementMatch]


@app.post("/match", response_model=MatchResponse)
def match(req: MatchRequest) -> MatchResponse:
    return run_match(req.cv_bullets, req.jd_text, req.jd_language, req.threshold, req.top_n)


def run_match(cv_bullets, jd_text, jd_language, threshold, top_n) -> MatchResponse:
    seg = pysbd.Segmenter(language=jd_language, clean=False)
    requirements = [s.strip() for s in seg.segment(jd_text)]

    provider = get_provider()
    bullet_vecs = provider.embed(cv_bullets)
    req_vecs = provider.embed(requirements)
    matrix = bullet_vecs @ req_vecs.T

    gaps = []
    requirement_matches = []
    for j, requirement in enumerate(requirements):
        col = matrix[:, j]
        if col.max() < threshold:
            gaps.append(requirement)
        top_idx = np.argsort(col)[::-1][:top_n]
        top = [BulletScore(bullet=cv_bullets[i], score=float(col[i])) for i in top_idx]
        requirement_matches.append(RequirementMatch(requirement=requirement, top_matches=top))


    return MatchResponse(
        requirement_sentences=requirements,
        cv_bullets=cv_bullets,
        matrix=matrix.tolist(),
        coverage_gaps=gaps,
        requirements=requirement_matches
    )
