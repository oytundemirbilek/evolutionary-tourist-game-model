"""Run the Monte Carlo simulation of the sharing economy trust game."""

from copy import copy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from touristgame.agents import Agent


import numpy as np

from touristgame.agents import MAX_REPUTATION, MIN_REPUTATION, Strategy
from touristgame.game import PayoffMatrix, SquareLatticeNetwork

STRATEGIES: tuple[Strategy, ...] = ("TP", "TC", "UP", "UC")

# Scenario 1: Equal proportions of the four strategies
# Scenario 2: More consumers than providers
# Scenario 3: More providers than consumers
# Scenario 4: More untrustworthy than trustworthy
# Scenario 5: More trustworthy than untrustworthy
SCENARIOS: dict[str, dict[Strategy, float]] = {
    "Scenario 1": {"TP": 0.25, "TC": 0.25, "UP": 0.25, "UC": 0.25},
    "Scenario 2": {"TP": 0.25, "TC": 0.25, "UP": 0.25, "UC": 0.25},
    "Scenario 3": {"TP": 0.25, "TC": 0.25, "UP": 0.25, "UC": 0.25},
    "Scenario 4": {"TP": 0.25, "TC": 0.25, "UP": 0.25, "UC": 0.25},
    "Scenario 5": {"TP": 0.25, "TC": 0.25, "UP": 0.25, "UC": 0.25},
}


class MonteCarloSimulation:
    """Run the game on a square lattice until strategy counts and total wealth stabilize."""

    def __init__(
        self,
        size: int = 100,
        scenario: str = "Scenario 1",
        reputation_threshold: float = 5.0,
        max_steps: int = 5000,
        n_last_steps_stability: int = 100,
        tolerance: float = 0.01,
    ) -> None:
        self.network = SquareLatticeNetwork(size)
        self.payoff_matrix = PayoffMatrix(reputation_threshold)
        self.agents = [agent for row in self.network.grid for agent in row]
        self.initial_proportions = SCENARIOS[scenario]

        self.max_steps = max_steps
        self.history: list[dict[str, float]] = []
        self.n_last_steps_stability = n_last_steps_stability

    def initialize(self) -> None:
        """Step 2: give each agent a random strategy, a random reputation and zero wealth."""
        strategies = list(self.initial_proportions)
        probabilities = list(self.initial_proportions.values())
        for agent in self.agents:
            agent.current_strategy = np.random.choice(strategies, p=probabilities)
            agent.current_wealth = 0.0
            agent.current_reputation = np.random.randint(
                MIN_REPUTATION, MAX_REPUTATION + 1
            )
        self.history.clear()

    def update_agent_payoffs(self, agent: "Agent") -> None:
        """Play a round for a single agent."""
        neighbors = self.network.get_neighbors(agent)
        agent.update_wealth(neighbors, self.payoff_matrix)
        agent.update_reputation()

    def update_agent_strategy(
        self, agent: "Agent", previous_network: "SquareLatticeNetwork"
    ) -> None:
        """Update a single agent's reputation and strategy."""
        # Get neighbor from the last timestep (before the neighbor's strategy is updated)
        neighbors = previous_network.get_neighbors(agent)
        neighbor = neighbors[np.random.randint(len(neighbors))]
        agent.update_strategy(neighbor.current_strategy, neighbor.current_wealth)

    def play_round(self) -> None:
        """Each agent plays its four neighbors and stores its summed payoff."""
        # Copy network to be able to get neighbor strategy from the last timestep.
        previous_network = copy(self.network)
        for agent in self.agents:
            # Step 3: Game Initiation
            self.update_agent_payoffs(agent)
            # Step 4: Strategy update
            self.update_agent_strategy(agent, previous_network)

    def calculate_total_revenue(self) -> float:
        """Calculate the total revenue of all agents."""
        return sum(agent.current_wealth for agent in self.agents)

    def save_current_state(self) -> None:
        """Step 5: store strategy counts, total wealth and mean reputation for this round."""
        self.history.append(
            {
                "total_wealth": sum(agent.current_wealth for agent in self.agents),
                "n_players_tp": sum(
                    1 for agent in self.agents if agent.current_strategy == "TP"
                ),
                "n_players_tc": sum(
                    1 for agent in self.agents if agent.current_strategy == "TC"
                ),
                "n_players_up": sum(
                    1 for agent in self.agents if agent.current_strategy == "UP"
                ),
                "n_players_uc": sum(
                    1 for agent in self.agents if agent.current_strategy == "UC"
                ),
            }
        )
