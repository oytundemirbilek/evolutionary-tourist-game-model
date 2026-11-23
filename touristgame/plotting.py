"""Plotting functions for the Tourist Game simulation."""

import matplotlib.pyplot as plt
import pandas as pd

PLAYER_COLUMNS = ["n_players_tp", "n_players_up", "n_players_tc", "n_players_uc"]


def plot_history(
    history: pd.DataFrame, show: bool = False, save_file_name: str | None = None
) -> None:
    """Plot the history of the simulation."""
    figure, (count_axes, wealth_axes) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    history[PLAYER_COLUMNS].plot(ax=count_axes, ylabel="Number of players")
    history["total_wealth"].plot(ax=wealth_axes, ylabel="Total wealth")
    figure.tight_layout()
    if save_file_name:
        figure.savefig(save_file_name, dpi=300)
    if show:
        plt.show()
