from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, status

from app import db
from app.auth import create_access_token, decode_access_token
from app.rag.global_rag import get_global_rag
from app.rag.index_builder import build_global_index
from app.rl_engine.feedback_handler import FeedbackHandler
from app.rl_engine.online_rlhf import OnlineRLHFTrainer
from app.rl_engine.reward_logger import log_feedback
from app.rl_engine.reward_model import rating_to_reward
from app.rl_engine.reward_tracker import RewardTracker
from app.rl_engine.response_modifier import ResponseModifier
from app.schemas import (
    AnalyticsResponse,
    AuthResponse,
    ChatRequest,
    ChatResponse,
    DocumentsResponse,
    FeedbackRequest,
    FeedbackResponse,
    LoginRequest,
    QualityGateRequest,
    QualityGateResponse,
    RebuildIndexResponse,
    RegisterRequest,
    RewardHistoryResponse,
)
from app.services.chat_service import generate_chat
from app.utils.guardrails import refresh_dynamic_keywords_cache


app = FastAPI(title="Reinforcement-Aligned Academic Intelligence System")


@app.on_event("startup")
def on_startup() -> None:
    db.init_db()
    refresh_dynamic_keywords_cache()


def _auth_error(detail: str = "Invalid or missing access token.") -> HTTPException:
    return HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)


def _forbidden(detail: str = "Forbidden") -> HTTPException:
    return HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=detail)


def get_current_user(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    if not authorization:
        raise _auth_error("Authorization header required.")
    parts = authorization.strip().split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise _auth_error("Authorization must be Bearer token.")

    token = parts[1].strip()
    try:
        payload = decode_access_token(token)
    except Exception:
        raise _auth_error()

    user = db.get_user(str(payload.get("sub", "")))
    if not user:
        raise _auth_error("User not found.")

    role = str(payload.get("role", "user")).lower()
    if role != str(user.get("role", "user")).lower():
        raise _auth_error("Role mismatch for token.")

    return {"user_id": str(user["user_id"]), "role": role}


def require_user(current_user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    if current_user["role"] != "user":
        raise _forbidden("Only user accounts can access this endpoint.")
    return current_user


def require_admin(current_user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    if current_user["role"] != "admin":
        raise _forbidden("Admin access required.")
    return current_user


@app.post("/auth/register", response_model=AuthResponse)
async def register(req: RegisterRequest):
    ok, message = db.create_user_credentials(req.user_id, req.password, role="user")
    if not ok:
        raise HTTPException(status_code=400, detail=message)

    token = create_access_token(message, "user")
    return AuthResponse(access_token=token, user_id=message, role="user")


@app.post("/auth/login", response_model=AuthResponse)
async def login(req: LoginRequest):
    user = db.authenticate_user(req.user_id, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials.")

    token = create_access_token(user["user_id"], user["role"])
    return AuthResponse(access_token=token, user_id=user["user_id"], role=user["role"])


@app.get("/auth/me")
async def me(current_user: dict[str, Any] = Depends(get_current_user)):
    return current_user


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, current_user: dict[str, Any] = Depends(require_user)):
    try:
        result = await generate_chat(
            user_id=current_user["user_id"],
            query=req.message,
            short_answer=req.short_answer,
            detailed_explanation=req.detailed_explanation,
        )
        return ChatResponse(**result)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat generation failed: {exc}")


@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(req: FeedbackRequest, current_user: dict[str, Any] = Depends(require_user)):
    reward = rating_to_reward(req.rating)
    log_feedback(
        chat_id=req.chat_id,
        user_id=current_user["user_id"],
        rating=req.rating,
        reward=reward,
        mode_used=req.mode_used or "detailed",
    )
    profile = FeedbackHandler.update_preferences(
        user_id=current_user["user_id"],
        reward=reward,
        mode_used=req.mode_used or "detailed",
    )
    question_profile = RewardTracker.update_from_feedback(req.chat_id, req.rating, reward)
    user_question_profile = RewardTracker.update_user_profile_from_feedback(
        req.chat_id,
        current_user["user_id"],
        req.rating,
        reward,
    )
    chat_row = db.get_chat_by_id(req.chat_id)
    OnlineRLHFTrainer.record_feedback_and_update_state(
        chat=chat_row,
        user_id=current_user["user_id"],
        rating=req.rating,
        reward=reward,
        question_profile=question_profile,
        user_question_profile=user_question_profile,
    )
    return FeedbackResponse(reward=reward, preference_profile=profile, question_profile=question_profile)


@app.post("/rebuild-index", response_model=RebuildIndexResponse)
async def rebuild_index(_: dict[str, Any] = Depends(require_admin)):
    try:
        out = build_global_index(force_rebuild=True)
        refresh_dynamic_keywords_cache()
        return RebuildIndexResponse(
            status=out.get("status", "ok"),
            total_pdfs=out.get("total_pdfs", 0),
            total_chunks=out.get("total_chunks", 0),
            documents=out.get("documents", []),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Index rebuild failed: {exc}")


@app.get("/documents", response_model=DocumentsResponse)
async def documents(_: dict[str, Any] = Depends(require_admin)):
    try:
        rag = get_global_rag()
        docs = rag.list_documents()
        return DocumentsResponse(
            total_pdfs=docs.get("total_pdfs", 0),
            total_chunks=docs.get("total_chunks", 0),
            documents=docs.get("documents", []),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Document listing failed: {exc}")


@app.get("/analytics", response_model=AnalyticsResponse)
async def analytics(
    user_id: str | None = None,
    scope: str = "all",
    weak_topics_limit: int = 1000,
    _: dict[str, Any] = Depends(require_admin),
):
    return AnalyticsResponse(**db.get_analytics(user_id=user_id, scope=scope, weak_topics_limit=weak_topics_limit))


@app.get("/reward-history", response_model=RewardHistoryResponse)
async def reward_history(limit: int = 100, user_id: str | None = None, _: dict[str, Any] = Depends(require_admin)):
    return RewardHistoryResponse(history=db.get_reward_history(limit=limit, user_id=user_id))


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/history")
async def history(limit: int = 50, _: dict[str, Any] = Depends(require_admin)):
    return db.get_history(limit=limit)


@app.get("/users")
async def users(limit: int = 500, _: dict[str, Any] = Depends(require_admin)):
    return {"users": db.list_users(limit=limit)}


@app.post("/quality-gate", response_model=QualityGateResponse)
async def quality_gate(req: QualityGateRequest, _: dict[str, Any] = Depends(require_admin)):
    out = ResponseModifier.quality_gate(req.query, req.response, mode=req.mode)
    return QualityGateResponse(**out)
