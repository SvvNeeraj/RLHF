from app import db


def log_feedback(chat_id: int, user_id: str, rating: int, reward: float, mode_used: str) -> None:
    db.save_feedback(chat_id=chat_id, user_id=user_id, rating=rating, reward=reward, mode_used=mode_used)
