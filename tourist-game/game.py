"""Define classes and functions related to the core game logic."""

from typing import Literal

import numpy as np

Strategy = Literal["TP", "TC", "UP", "UC"]
reward, sucker_payoff, temptation, punishment = 3, 1, 5, 2
BASE_PAYOFF: dict[tuple[Strategy, Strategy], tuple[float, float]] = {
    ("TP", "TC"): (reward, reward),
    ("TP", "UC"): (-sucker_payoff, temptation),
    ("UP", "TC"): (punishment, -punishment),
    ("UP", "UC"): (punishment, -punishment),
}


class SquareLatticeNetwork:
    """Represent a square lattice network."""

    def __init__(self, size: int = 100) -> None:
        self.size = size  # L
        self.grid = np.zeros((size, size), dtype=int)  # 10000 agents


class PayoffMatrix:
    """Represent a payoff matrix."""

    def __init__(self, reputation_threshold: float = 50.0) -> None:

        self.reputation_threshold = reputation_threshold  # Z
        self.base_matrix = BASE_PAYOFF

    def calculate_reputation_multiplier(self, reputation_score: float) -> float:
        """Calculate the reputation multipleier based on reputation score and threshold."""
        return (
            reputation_score - self.reputation_threshold
        ) / self.reputation_threshold

    def get_payoff(
        self, strategy1: Strategy, strategy2: Strategy, reputation_score: float
    ) -> tuple[float, float] | None:
        """Get the payoff resulting from the interaction of two given strategies."""
        multiplier = 1 + self.calculate_reputation_multiplier(reputation_score)
        scores = self.base_matrix.get((strategy1, strategy2))
        if scores is None:
            return None
        score1, score2 = scores
        return score1 * multiplier, score2 * multiplier


class ReputationScoreMatrix:
    """Represent a reputation score matrix."""

    def __init__(self) -> None:
        self.reputation_scoring_matrix = BASE_PAYOFF

    def get_reputation(
        self, strategy1: Strategy, strategy2: Strategy
    ) -> tuple[float, float] | None:
        """Get the reputation score resulting from the interaction of two given strategies."""
        return self.reputation_scoring_matrix.get((strategy1, strategy2))


if __name__ == "__main__":
    # Example usage
    reputation_matrix = ReputationScoreMatrix()
    scores = reputation_matrix.get_reputation("TP", "TC")
    print(scores)
