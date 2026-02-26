
def rating_to_reward(rating: int) -> float:
    # Maps 1..5 to -1.0..1.0
    return (rating - 3) / 2
