"""Define agents and their actions related to the core game logic."""

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from game import PayoffMatrix

import numpy as np

Strategy = Literal["TP", "TC", "UP", "UC"]
MIN_REPUTATION = 0
MAX_REPUTATION = 20
TRUSTWORTHY_STRATEGIES = {"TP", "TC"}
PROVIDER_STRATEGIES = {"TP", "UP"}


class Agent:
    """Represent an internal agent in the game."""

    def __init__(
        self,
        location: tuple[int, int],
        initial_strategy: Strategy = "TC",
        initial_reputation: float = 0.0,
        initial_wealth: float = 0.0,
    ) -> None:
        self.location = location
        self.current_strategy: Strategy = initial_strategy
        self.current_reputation = initial_reputation
        self.current_wealth = initial_wealth

    def calculate_strategy_update_probability(self, neighbor_wealth: float) -> float:
        """Update the agent's strategy based on its current strategy."""
        fermi_term = 1.0 / (1.0 + np.exp(self.current_wealth - neighbor_wealth)) - 0.5
        return np.abs(fermi_term) + fermi_term

    def update_strategy(
        self, neighbor_strategy: Strategy, neighbor_wealth: float
    ) -> None:
        """Update the agent's strategy based on its current strategy."""
        update_probability = self.calculate_strategy_update_probability(neighbor_wealth)
        if np.random.rand() < update_probability:
            self.current_strategy = neighbor_strategy

    def update_reputation(self, reputation_change: float = 1.0) -> None:
        """Update the agent's reputation based on the reputation change."""
        if self.current_strategy in TRUSTWORTHY_STRATEGIES:
            self.current_reputation += reputation_change
        else:
            self.current_reputation -= reputation_change
        self.current_reputation = max(
            MIN_REPUTATION, min(MAX_REPUTATION, self.current_reputation)
        )

    def update_wealth(
        self, neighbors: list["Agent"], payoff_matrix: "PayoffMatrix"
    ) -> None:
        """Set the agent's net wealth from its interactions with its neighbors (Eq. 1)."""
        total_wealth = 0.0
        for neighbor in neighbors:
            if self.current_strategy in PROVIDER_STRATEGIES:
                strategy1, strategy2 = self.current_strategy, neighbor.current_strategy
                own_index = 0
            else:
                strategy1, strategy2 = neighbor.current_strategy, self.current_strategy
                own_index = 1
            payoffs = payoff_matrix.get_payoff(
                strategy1, strategy2, self.current_reputation
            )

            if payoffs is not None:
                total_wealth += payoffs[own_index]

        self.current_wealth += total_wealth


class ExternalAgent:
    """Represent an external agent in the game."""

    def __init__(self, initial_strategy: Strategy = "TC") -> None:
        self.strategy = initial_strategy
