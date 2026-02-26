from app import db


class RewardTracker:
    @staticmethod
    def get_query_profile(query: str, user_id: str | None = None) -> dict:
        return db.get_question_reward(query, user_id=user_id)

    @staticmethod
    def update_from_feedback(chat_id: int, rating: int, reward: float) -> dict:
        chat = db.get_chat_by_id(chat_id)
        if not chat:
            return {
                "query_key": "",
                "sample_question": "",
                "avg_reward": reward,
                "total_feedback": 1,
                "low_rating_count": 1 if rating <= 2 else 0,
                "last_rating": rating,
            }
        return db.upsert_question_reward(
            query_key=chat["query_key"],
            sample_question=chat["query"],
            rating=rating,
            reward=reward,
        )

    @staticmethod
    def update_user_profile_from_feedback(chat_id: int, user_id: str, rating: int, reward: float) -> dict:
        chat = db.get_chat_by_id(chat_id)
        if not chat:
            return {
                "user_id": user_id,
                "query_key": "",
                "sample_question": "",
                "avg_reward": reward,
                "total_feedback": 1,
                "low_rating_count": 1 if rating <= 2 else 0,
                "last_rating": rating,
            }
        return db.upsert_question_reward_user(
            user_id=user_id,
            query_key=chat["query_key"],
            sample_question=chat["query"],
            rating=rating,
            reward=reward,
        )
