"""Main entrypoint for the Tourist Game simulation."""

import multiprocessing

import numpy as np
import pandas as pd

from touristgame.montecarlo import MonteCarloSimulation
from touristgame.plotting import plot_history

# the reputation threshold Z is set to 2, 6, and 12


class Experiment(multiprocessing.Process):
    """Run the game on a square lattice in parallel for different parameters."""

    def __init__(self, p_id: int, scenario: str, reputation_threshold: int) -> None:
        super().__init__()
        self.p_id = p_id
        self.scenario = scenario
        self.reputation_threshold = reputation_threshold

    def run(self, plot: bool = True) -> None:
        """Run the Monte Carlo simulation for the given parameters."""
        simulator = MonteCarloSimulation(
            size=100,
            scenario=self.scenario,
            reputation_threshold=self.reputation_threshold,
            max_steps=10000,
            snapshot_steps=(9, 29, 99, 9999),
        )
        # evolution process of different strategies over time steps t = 10, 30, 100, and 10,000

        # pass the process ID to set the position of the progress bar
        simulator.run(self.p_id)
        history_df = simulator.history_to_dataframe()
        snapshot_matrix = simulator.snapshots_to_matrix()
        np.save(
            f"results/simulation_snapshots_{self.scenario}_Z{self.reputation_threshold}.npy",
            snapshot_matrix,
        )
        history_df.to_csv(
            f"results/simulation_history_{self.scenario}_Z{self.reputation_threshold}.csv",
            index=True,
        )  # index is the step
        # history_df = pd.read_csv(
        #     "simulation_history_scenario1.csv", index_col=0
        # )  # index is the step
        if plot:
            plot_history(
                history=history_df,
                save_file_name=f"results/simulation_history_{self.scenario}_Z{self.reputation_threshold}.png",
                show=False,
            )


if __name__ == "__main__":
    Z = [2, 4, 6, 8, 10]
    processes = []
    for i, z in enumerate(Z):
        p = Experiment(i, "scenario1", z)
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    for p in processes:
        p.terminate()

    Z = [12, 14, 16, 18, 20]
    processes = []
    for i, z in enumerate(Z):
        p = Experiment(i, "scenario1", z)
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    for p in processes:
        p.terminate()

    # p0 = Experiment(0, "scenario1", 2)
    # p1 = Experiment(1, "scenario1", 6)
    # p0.start()
    # p1.start()
    # p0.join()
    # p1.join()
