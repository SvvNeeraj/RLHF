import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import settings

_LOCK = threading.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    _ensure_parent(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    _ensure_parent(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    out = []
    for p in parts:
        s = " ".join(p.split()).strip()
        if len(s) < 30:
            continue
        out.append(s)
    return out


def _extract_snippets(text: str, max_items: int = 3) -> list[str]:
    out = []
    for s in _sentences(text):
        cleaned = re.sub(r"[^\x20-\x7E]", " ", s)
        cleaned = " ".join(cleaned.split()).strip()
        if 30 <= len(cleaned) <= 180:
            out.append(cleaned)
        if len(out) >= max_items:
            break
    return out


class OnlineRLHFTrainer:
    @staticmethod
    def _state_key(user_id: str, query_key: str) -> str:
        return f"{(user_id or '').strip().lower()}::{(query_key or '').strip().lower()}"

    @staticmethod
    def record_feedback_and_update_state(
        chat: dict[str, Any] | None,
        user_id: str,
        rating: int,
        reward: float,
        question_profile: dict[str, Any],
        user_question_profile: dict[str, Any] | None = None,
    ) -> None:
        if not chat:
            return

        event = {
            "timestamp": _utc_now(),
            "user_id": user_id,
            "chat_id": chat.get("id"),
            "query": chat.get("query", ""),
            "query_key": chat.get("query_key", ""),
            "response": chat.get("response", ""),
            "rating": int(rating),
            "reward": float(reward),
            "mode_used": chat.get("mode_used", ""),
            "request_style": chat.get("request_style", ""),
            "question_profile": question_profile or {},
            "user_question_profile": user_question_profile or {},
        }

        signal = "neutral"
        if rating >= 4:
            signal = "positive"
        elif rating <= 2:
            signal = "negative"

        train_record = {
            "timestamp": event["timestamp"],
            "signal": signal,
            "user_id": user_id,
            "query": event["query"],
            "query_key": event["query_key"],
            "response": event["response"],
            "rating": event["rating"],
            "reward": event["reward"],
            "request_style": event["request_style"],
            "dynamic_training_note": (
                "Use as preferred target style for similar future queries."
                if signal == "positive"
                else "Avoid phrasing/content style; regenerate with clearer wording and different examples."
                if signal == "negative"
                else "Informative sample."
            ),
        }

        feedback_path = Path(settings.rlhf_feedback_jsonl_path)
        training_path = Path(settings.rlhf_training_jsonl_path)
        state_path = Path(settings.rlhf_state_json_path)

        with _LOCK:
            _append_jsonl(feedback_path, event)
            _append_jsonl(training_path, train_record)

            state = _load_json(state_path, default={"profiles": {}, "updated_at": ""})
            key = OnlineRLHFTrainer._state_key(user_id=user_id, query_key=str(chat.get("query_key", "")))
            cur = state.get("profiles", {}).get(
                key,
                {
                    "user_id": user_id,
                    "query_key": chat.get("query_key", ""),
                    "sample_query": chat.get("query", ""),
                    "total_feedback": 0,
                    "avg_reward": 0.0,
                    "low_rating_count": 0,
                    "high_rating_count": 0,
                    "last_rating": 0,
                    "avoid_snippets": [],
                    "preferred_snippets": [],
                    "updated_at": "",
                },
            )

            n = int(cur.get("total_feedback", 0))
            avg = float(cur.get("avg_reward", 0.0))
            new_n = n + 1
            new_avg = ((avg * n) + float(reward)) / new_n

            cur["total_feedback"] = new_n
            cur["avg_reward"] = round(new_avg, 6)
            cur["last_rating"] = int(rating)
            cur["updated_at"] = event["timestamp"]
            cur["sample_query"] = chat.get("query", cur.get("sample_query", ""))

            if int(rating) <= 2:
                cur["low_rating_count"] = int(cur.get("low_rating_count", 0)) + 1
                snippets = _extract_snippets(chat.get("response", ""), max_items=3)
                combined = list(cur.get("avoid_snippets", []))
                for s in snippets:
                    if s not in combined:
                        combined.append(s)
                cur["avoid_snippets"] = combined[-12:]
            elif int(rating) >= 4:
                cur["high_rating_count"] = int(cur.get("high_rating_count", 0)) + 1
                snippets = _extract_snippets(chat.get("response", ""), max_items=2)
                combined = list(cur.get("preferred_snippets", []))
                for s in snippets:
                    if s not in combined:
                        combined.append(s)
                cur["preferred_snippets"] = combined[-10:]

            state.setdefault("profiles", {})[key] = cur
            state["updated_at"] = event["timestamp"]
            _save_json(state_path, state)

    @staticmethod
    def get_adaptation_plan(user_id: str, query_key: str) -> dict[str, Any]:
        state_path = Path(settings.rlhf_state_json_path)
        key = OnlineRLHFTrainer._state_key(user_id=user_id, query_key=query_key)
        with _LOCK:
            state = _load_json(state_path, default={"profiles": {}})
        row = state.get("profiles", {}).get(key, {})
        if not row:
            return {
                "exists": False,
                "total_feedback": 0,
                "avg_reward": 0.0,
                "low_rating_count": 0,
                "avoid_snippets": [],
                "preferred_snippets": [],
            }
        return {
            "exists": True,
            "total_feedback": int(row.get("total_feedback", 0)),
            "avg_reward": float(row.get("avg_reward", 0.0)),
            "low_rating_count": int(row.get("low_rating_count", 0)),
            "avoid_snippets": list(row.get("avoid_snippets", [])),
            "preferred_snippets": list(row.get("preferred_snippets", [])),
            "last_rating": int(row.get("last_rating", 0)),
        }

