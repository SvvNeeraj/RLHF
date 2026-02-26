from app import db


class FeedbackHandler:
    MODES = ("short", "detailed")

    @staticmethod
    def update_preferences(user_id: str, reward: float, mode_used: str) -> dict:
        current = db.get_user_pref(user_id)
        n = current["total_feedback"]
        avg = current["avg_reward"]

        new_n = n + 1
        new_avg = ((avg * n) + reward) / new_n

        preferred_mode = current["preferred_mode"]
        if reward > 0.2 and mode_used in FeedbackHandler.MODES:
            preferred_mode = mode_used
        elif reward < -0.4:
            preferred_mode = "detailed"

        db.upsert_user_pref(
            user_id=user_id,
            preferred_mode=preferred_mode,
            avg_reward=new_avg,
            total_feedback=new_n,
        )

        return {
            "preferred_mode": preferred_mode,
            "avg_reward": round(new_avg, 4),
            "total_feedback": new_n,
        }
