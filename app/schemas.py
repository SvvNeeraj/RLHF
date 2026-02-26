from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    message: str
    short_answer: bool = False
    detailed_explanation: bool = True


class ChatResponse(BaseModel):
    chat_id: int
    response: str
    rejected: bool = False
    sources: list[dict] = []
    response_improved: bool = False
    improvement_reason: str = ""
    response_variant: str = ""


class FeedbackRequest(BaseModel):
    chat_id: int
    rating: int = Field(ge=1, le=5)
    mode_used: Optional[str] = "detailed"


class FeedbackResponse(BaseModel):
    reward: float
    preference_profile: dict
    question_profile: dict


class RebuildIndexResponse(BaseModel):
    status: str
    total_pdfs: int
    total_chunks: int
    documents: list[dict]


class DocumentsResponse(BaseModel):
    total_pdfs: int
    total_chunks: int
    documents: list[dict]


class AnalyticsResponse(BaseModel):
    total_chats: int
    total_feedback: int
    avg_rating: float
    avg_reward: float
    weak_topics: list[dict]


class RewardHistoryResponse(BaseModel):
    history: list[dict]


class HistoryResponse(BaseModel):
    chats: list[dict]
    feedback: list[dict]


class QualityGateRequest(BaseModel):
    query: str
    response: str
    mode: str = Field(default="detailed")


class QualityGateResponse(BaseModel):
    passed: bool
    reasons: list[str]


class RegisterRequest(BaseModel):
    user_id: str
    password: str


class LoginRequest(BaseModel):
    user_id: str
    password: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    role: str
