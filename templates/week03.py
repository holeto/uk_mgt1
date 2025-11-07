#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from templates.week01 import calculate_best_response_against_row, calculate_best_response_against_col, evaluate_general_sum

def compute_deltas(
    row_matrix: np.ndarray,
    col_matrix: np.ndarray,
    row_strategy: np.ndarray,
    col_strategy: np.ndarray,
) -> np.ndarray:
    """Compute players' incentives to deviate from their strategies.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    row_strategy : np.ndarray
        The row player's strategy
    col_strategy : np.ndarray
        The column player's strategy

    Returns
    -------
    np.ndarray
        Each player's incentive to deviate
    """

    col_br = calculate_best_response_against_row(col_matrix, row_strategy)
    row_br = calculate_best_response_against_col(row_matrix, col_strategy)

    row_br_val = np.sum(row_matrix[row_br.astype(bool)].squeeze(0) * col_strategy)
    col_br_val = np.sum(col_matrix[:, col_br.astype(bool)].squeeze(-1) * row_strategy)

    row_val, col_val = evaluate_general_sum(row_matrix, col_matrix, row_strategy, col_strategy)
    return np.array([row_br_val - row_val, col_br_val - col_val])


def compute_nash_conv(
    row_matrix: np.ndarray,
    col_matrix: np.ndarray,
    row_strategy: np.ndarray,
    col_strategy: np.ndarray,
) -> float:
    """Compute the NashConv value of a given strategy profile.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    row_strategy : np.ndarray
        The row player's strategy
    col_strategy : np.ndarray
        The column player's strategy

    Returns
    -------
    float
        The NashConv value of the given strategy profile
    """

    incentives_to_deviate = compute_deltas(row_matrix, col_matrix, row_strategy, col_strategy)
    return np.sum(incentives_to_deviate)


def compute_exploitability(
    row_matrix: np.ndarray,
    col_matrix: np.ndarray,
    row_strategy: np.ndarray,
    col_strategy: np.ndarray,
) -> float:
    """Compute the exploitability of a given strategy profile.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    row_strategy : np.ndarray
        The row player's strategy
    col_strategy : np.ndarray
        The column player's strategy

    Returns
    -------
    float
        The exploitability value of the given strategy profile
    """

    num_players = 2
    return compute_nash_conv(row_matrix, col_matrix, row_strategy, col_strategy) / num_players


def fictitious_play(
    row_matrix: np.ndarray, col_matrix: np.ndarray, num_iters: int, naive: bool
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Run Fictitious Play for a given number of epochs.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    num_iters : int
        The number of iterations to run the algorithm for
    naive : bool
        Whether to calculate the best response against the last
        opponent's strategy or the average opponent's strategy

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        The sequence of average strategy profiles produced by the algorithm
    """

    found_strategies = []

    num_players = 2
    row_actions = row_matrix.shape[0]
    col_actions = col_matrix.shape[1]
    max_actions = max(row_actions, col_actions)
    per_player_actions = np.array([row_actions, col_actions])[..., None]
    valid_action_indices = np.tile(np.arange(max_actions), (num_players, 1)) < per_player_actions

    #action_frequencies = np.zeros((num_players, max_actions), dtype=np.int32) + valid_action_indices
    #For now we initialize as the best response 
    # to the uniform strategy of oponent
    row_uniform_br = np.pad(calculate_best_response_against_col(row_matrix, np.ones(col_actions) / col_actions), (0, max_actions - row_actions), constant_values=0)
    col_uniform_br = np.pad(calculate_best_response_against_row(col_matrix, np.ones(row_actions) / row_actions), (0, max_actions - col_actions), constant_values=0)
    last_played = np.stack([row_uniform_br, col_uniform_br], axis=0)
    #avg_strategies = last_played.copy()
    #uniform initialization
    avg_strategies = valid_action_indices / valid_action_indices.sum(axis=-1, keepdims=True)
    #print(f"Init action frequncies {action_frequencies}, row_actions {row_actions}, col_action {col_actions}")
    for i in range(num_iters):
        #action_frequencies = action_frequencies + last_played
        avg_strategies = ((1 - (1 / (i + 1))) * avg_strategies + (1 / (i + 1)) * last_played)
        #avg_strategies = action_frequencies / np.sum(action_frequencies, axis=-1, keepdims=True)
        assert np.all(avg_strategies[0, row_actions:] == 0), f"Invalid action for row player was played. Row actions {row_actions}, frequencies {avg_strategies[0]}"
        assert np.all(avg_strategies[1, col_actions:] == 0), f"Invalid action for row player was played. Col actions {col_actions}, frequencies {avg_strategies[1]}"
        if naive:
            cur_strategies = last_played
        else:
            cur_strategies = avg_strategies
        row_br = np.pad(calculate_best_response_against_col(row_matrix, cur_strategies[1][:col_actions]), (0, max_actions - row_actions), constant_values=0)
        col_br = np.pad(calculate_best_response_against_row(col_matrix, cur_strategies[0][:row_actions]), (0, max_actions - col_actions), constant_values=0)
        last_played = np.stack([row_br, col_br], axis=0)
        # print(f"Iteration: {i}")
        # print(f"Frequencies: {action_frequencies}")
        # print(f"Best responding to {cur_strategies}")
        # print(f"Current brs: {last_played}")
        #We always want to return the average strategy, whether
        # doing the naive algorithm or the proper one
        found_strategies.append((avg_strategies[0, :row_actions], avg_strategies[1, :col_actions]))
    return found_strategies    


def plot_exploitability(
    row_matrix: np.ndarray,
    col_matrix: np.ndarray,
    strategies: list[tuple[np.ndarray, np.ndarray]],
    label: str,
) -> list[float]:
    """Compute and plot the exploitability of a sequence of strategy profiles.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    strategies : list[tuple[np.ndarray, np.ndarray]]
        The sequence of strategy profiles
    label : str
        The name of the algorithm that produced `strategies`

    Returns
    -------
    list[float]
        A sequence of exploitability values, one for each strategy profile
    """

    exploitabilities = []
    iterations = np.arange(len(strategies))
    for strategy in strategies:
        exp = compute_exploitability(row_matrix, col_matrix, strategy[0], strategy[1])
        exploitabilities.append(exp)
    exploitabilities = np.asarray(exploitabilities)
    fig, ax = plt.subplots()
    plt.title(label=label)
    plt.plot(iterations, exploitabilities)
    plt.xlabel("Iterations")
    plt.ylabel("Exploitability")
    plt.show()
    plt.close(fig)
    return exploitabilities

def main() -> None:
    num_iters = 200
    matching_pennies= np.array([[1, -1], [-1, 1]])
    fp_mp_strategies = fictitious_play(matching_pennies, -matching_pennies, num_iters, False)
    expls = plot_exploitability(matching_pennies, -matching_pennies, fp_mp_strategies, "Fictitous play Matching Pennies")
    #breakpoint()
    fp_mp_naive_strategies = fictitious_play(matching_pennies, -matching_pennies, num_iters, True)
    expls = plot_exploitability(matching_pennies, -matching_pennies, fp_mp_naive_strategies, "Naive Fictitous play Matching Pennies")
    #breakpoint()
    rps = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
    fp_rps_strategies = fictitious_play(rps, -rps, num_iters, False)
    expls = plot_exploitability(rps, -rps, fp_rps_strategies, "Fictitous play RPS")
    #breakpoint()
    fp_rps_naive_strategies = fictitious_play(rps, -rps, num_iters, True)
    expls = plot_exploitability(rps, -rps, fp_rps_naive_strategies, "Naive Fictitous play RPS")
    #breakpoint()
    battle_of_sexes_row = np.array([[3, 0], [0, 2]])
    battle_of_sexes_col = np.array([[2, 0], [0, 3]])
    fp_bos_strategies = fictitious_play(battle_of_sexes_row, battle_of_sexes_col, num_iters, False)
    expls = plot_exploitability(battle_of_sexes_row, battle_of_sexes_col, fp_bos_strategies, "Fictitous play Battle of Sexes")
    #breakpoint()
    fp_bos_naive_strategies = fictitious_play(battle_of_sexes_row, battle_of_sexes_col, num_iters, True)
    expls = plot_exploitability(battle_of_sexes_row, battle_of_sexes_col, fp_bos_naive_strategies, "Naive Fictitous play Battle of Sexes")
    #breakpoint()
    prisoner_dillema_row = np.array([[1, 3], [0, 2]])
    prisoner_dillema_col = np.array([[1, 0], [3, 2]])
    fp_pd_strategies = fictitious_play(prisoner_dillema_row, prisoner_dillema_col, num_iters, False)
    expls = plot_exploitability(prisoner_dillema_row, prisoner_dillema_col, fp_pd_strategies, "Fictitous play Prisoner's Dilemma")
    #breakpoint()
    fp_pd_naive_strategies = fictitious_play(prisoner_dillema_row, prisoner_dillema_col, num_iters, True)
    expls = plot_exploitability(prisoner_dillema_row, prisoner_dillema_col, fp_pd_naive_strategies, "Naive Fictitous play Prisoner's Dilemma")
    #breakpoint()

if __name__ == '__main__':
    main()
