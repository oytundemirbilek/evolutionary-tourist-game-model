"""Define agents and their actions related to the core game logic."""


class Agent:
    """Represent an internal agent in the game."""

    def __init__(
        self,
        location: tuple[int, int],
        initial_strategy: str = "random",
        initial_reputation: float = 0.5,
    ) -> None:
        self.location = location
        self.strategy = initial_strategy
        self.reputation = initial_reputation

    def strategy_update(self) -> None:
        """Update the agent's strategy based on its current strategy."""
        # Placeholder
        return


class ExternalAgent:
    """Represent an external agent in the game."""

    def __init__(self, initial_strategy: str = "cooperate") -> None:
        self.strategy = initial_strategy
