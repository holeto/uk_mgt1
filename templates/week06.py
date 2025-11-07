#!/usr/bin/env python3

import numpy as np
from templates.week03 import plot_exploitability, compute_exploitability


def regret_matching(regrets: np.ndarray) -> np.ndarray:
    """Generate a strategy based on the given cumulative regrets.

    Parameters
    ----------
    regrets : np.ndarray
        The vector containing cumulative regret of each action

    Returns
    -------
    np.ndarray
        The generated strategy
    """

    positive_regrets = np.maximum(regrets, 0)
    N = positive_regrets.shape[0]
    normalization = np.sum(positive_regrets)
    strategy = positive_regrets / (normalization + (normalization == 0))
    strategy = np.where(normalization > 0, strategy, 1 / N)
    return strategy


def regret_minimization(
    row_matrix: np.ndarray, col_matrix: np.ndarray, num_iters: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Run Regret Minimization for a given number of iterations.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    num_iters : int
        The number of iterations to run the algorithm for

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        The sequence of `num_iters` average strategy profiles produced by the algorithm
    """

    R = row_matrix.shape[0]
    C = col_matrix.shape[1]

    #uniform initialization
    row_avg_strategy = np.ones(R) / R
    col_avg_strategy = np.ones(C) / C

    #Also mantain cumulative regrets.
    # CRITICAL! Use the cumulative
    # regret to update the strategies
    # and NOT just current regret
    row_cum_regrets = np.zeros(R)
    col_cum_regrets = np.zeros(C)

    strategies = []

    for i in range(num_iters):
        row_cur_strategy = regret_matching(row_cum_regrets)
        col_cur_strategy = regret_matching(col_cum_regrets)

        row_action_values = row_matrix @ col_cur_strategy
        row_value = np.sum(row_action_values * row_cur_strategy)
        row_cum_regrets += row_action_values - row_value

        col_action_values = col_matrix.T @ row_cur_strategy
        col_value = np.sum(col_action_values * col_cur_strategy)
        col_cum_regrets += col_action_values - col_value

        weight = 1 / (i + 1)
        row_avg_strategy = (1 - weight) * row_avg_strategy +  weight * row_cur_strategy
        col_avg_strategy = (1 - weight) * col_avg_strategy + weight * col_cur_strategy
        strategies.append((row_avg_strategy, col_avg_strategy))
    return strategies


def main() -> None:
    unique_pure_ne_game = np.array([[1, 2], [0, -1]])
    strategies = regret_minimization(unique_pure_ne_game, -unique_pure_ne_game, 100)
    plot_exploitability(unique_pure_ne_game, -unique_pure_ne_game, strategies, "Regret matching exploitability")


if __name__ == '__main__':
    main()
