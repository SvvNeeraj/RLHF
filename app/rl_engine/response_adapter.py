
class ResponseAdapter:
    @staticmethod
    def choose_style(avg_reward: float, preferred_mode: str, short_answer: bool, detailed_explanation: bool) -> tuple[str, str]:
        # mode tracks preferred explanation length from feedback trend
        mode = preferred_mode
        if short_answer:
            mode = "short"
        elif detailed_explanation:
            mode = "detailed"

        if mode == "short":
            base = "Answer in concise bullet points and include only key exam facts."
        else:
            base = "Explain concepts step-by-step with short examples and exam-focused structure."

        quality_hint = (
            "User ratings are high; maintain strong structure and precision."
            if avg_reward >= 0.4
            else "User ratings are mixed; keep language simple and avoid unnecessary detail."
        )
        return mode, f"{base} {quality_hint}"
