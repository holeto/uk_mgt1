#!/usr/bin/env python3

from templates.week04 import find_nash_equilibrium
from templates.week03 import compute_exploitability
from templates.week01 import calculate_best_response_against_col, calculate_best_response_against_row, evaluate_zero_sum
import numpy as np


def double_oracle(
    row_matrix: np.ndarray, eps: float, rng: np.random.Generator
) -> tuple[list[np.ndarray, np.ndarray], list[np.ndarray, np.ndarray]]:
    """Run Double Oracle until a termination condition is met.

    The reference implementation generates the initial restricted game by
    randomly sampling one pure action for each player using `rng.integers`.

    The algorithm terminates when either:
        1. the difference between the upper and the lower bound on the game value drops below `eps`
        2. both players' best responses are already contained in the current restricted game

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    eps : float
        The required accuracy for the approximate Nash equilibrium
    rng : np.random.Generator
        A random number generator

    Returns
    -------
    tuple[list[np.ndarray, np.ndarray], list[np.ndarray, np.ndarray]]
        A tuple containing a sequence of strategy profiles and a sequence of corresponding supports
    """
    R = row_matrix.shape[0]
    C = row_matrix.shape[1]
    row_actions = np.arange(R)
    col_actions = np.arange(C)

    row_in_support = np.zeros(R, dtype=np.bool)
    col_in_support = np.zeros(C, dtype=np.bool)
    row_new_action = rng.integers(0, R)
    col_new_action = rng.integers(0, C)

    row_support_size = 0
    col_support_size = 0
    
    supports = []

    #Just saving it to return the supports that
    # are ordered by the algorithm run, eg.
    # as the actions were being added to the supports
    row_supports = []
    col_supports = []

    strategies = []

    while True:
        #Add the last selected actions
        #Only increase the support size if they are not in support already
        if not row_in_support[row_new_action]:
            row_support_size += 1
            row_supports.append(row_new_action)
            row_in_support[row_new_action] = True
        if not col_in_support[col_new_action]:
            col_support_size += 1
            col_supports.append(col_new_action)
            col_in_support[col_new_action] = True
        row_support_indices = np.nonzero(row_in_support)[0].astype(np.int64)
        col_support_indices = np.nonzero(col_in_support)[0].astype(np.int64)
        #Handle the edge cases where both players have only single action in support
        if row_support_size == 1 and col_support_size == 1:
            row_full_game_strategy = row_in_support.astype(np.float64)
            col_full_game_strategy = col_in_support.astype(np.float64)
        else:
            subgame = row_matrix[row_support_indices[..., None], col_support_indices]
            row_nash, col_nash = find_nash_equilibrium(subgame)
            #Perform the remapping
            row_full_game_strategy = np.zeros(R, dtype=np.float64)
            row_full_game_strategy[row_support_indices] = row_nash
            col_full_game_strategy = np.zeros(C, dtype=np.float64)
            col_full_game_strategy[col_support_indices] = col_nash

        supports.append((np.array(row_supports, dtype=np.int64), np.array(col_supports, dtype=np.int64)))
        strategies.append((row_full_game_strategy, col_full_game_strategy))
        row_br = calculate_best_response_against_col(row_matrix, col_full_game_strategy)
        col_br = calculate_best_response_against_row(-row_matrix, row_full_game_strategy)
        #Check if the best responses are in the game already for both_players
        # if yes, terminate
        row_new_action = np.sum(row_br * row_actions).astype(np.int64)
        col_new_action = np.sum(col_br * col_actions).astype(np.int64)
        if row_in_support[row_new_action] and col_in_support[col_new_action]:
            break
        #lower bound on the value is if the player
        # sticks to the strategy and opponent best responds,
        # and upper bound is if opponent sticks to the strategy and
        # the player best responds
        v_lb = evaluate_zero_sum(row_matrix, row_full_game_strategy, col_br)[0]
        v_ub = evaluate_zero_sum(row_matrix, row_br, col_full_game_strategy)[0]
        #Termination if the values are sufficiently close
        if v_ub - v_lb <= eps:
            break
    return strategies, supports

def main() -> None:
    rps = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
    rng = np.random.default_rng(seed=42)
    rps_strategies, rps_supports = double_oracle(rps, 0, rng)
    print(f"RPS strategies {rps_strategies}")
    print(f"RPS supports {rps_supports}")
    print(f"RPS exploitability: {compute_exploitability(rps, -rps, *rps_strategies[-1])}")
    mp = np.array([[1, -1], [-1, 1]])
    mp_strategies, mp_supports = double_oracle(mp, 0, rng)
    print(f"MP strategies {mp_strategies}")
    print(f"MP supports {mp_supports}")
    print(f"MP exploitability: {compute_exploitability(mp, -mp, *mp_strategies[-1])}")


if __name__ == '__main__':
    main()
