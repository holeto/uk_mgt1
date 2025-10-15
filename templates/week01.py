#!/usr/bin/env python3

import numpy as np


def evaluate_general_sum(
    row_matrix: np.ndarray,
    col_matrix: np.ndarray,
    row_strategy: np.ndarray,
    col_strategy: np.ndarray,
) -> np.ndarray:
    """Compute the expected utility of each player in a general-sum game.

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
        A vector of expected utilities of the players
    """

    #Broadcast the row strategy over the columns
    # and the column strategy over the rows
    p1_util = np.sum(row_matrix * row_strategy[..., None] * col_strategy[None, ...])
    p2_util = np.sum(col_matrix * row_strategy[..., None] * col_strategy[None, ...])
    return np.asarray([p1_util, p2_util])


def evaluate_zero_sum(
    row_matrix: np.ndarray, row_strategy: np.ndarray, col_strategy: np.ndarray
) -> np.ndarray:
    """Compute the expected utility of each player in a zero-sum game.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    row_strategy : np.ndarray
        The row player's strategy
    col_strategy : np.ndarray
        The column player's strategy

    Returns
    -------
    np.ndarray
        A vector of expected utilities of the players
    """

    p1_util = np.sum(row_matrix * row_strategy[..., None] * col_strategy[None, ...])
    return np.asarray([p1_util, -p1_util])


def calculate_best_response_against_row(
    col_matrix: np.ndarray, row_strategy: np.ndarray
) -> np.ndarray:
    """Compute a pure best response for the column player against the row player.

    Parameters
    ----------
    col_matrix : np.ndarray
        The column player's payoff matrix
    row_strategy : np.ndarray
        The row player's strategy

    Returns
    -------
    np.ndarray
        The column player's best response
    """

    col_action_values = np.sum(col_matrix * row_strategy[..., None], axis=0)

    col_br_action = np.argmax(col_action_values)
    
    col_actions = col_matrix.shape[1]

    #This handles one hot encoding
    return np.eye(col_actions)[col_br_action]


def calculate_best_response_against_col(
    row_matrix: np.ndarray, col_strategy: np.ndarray
) -> np.ndarray:
    """Compute a pure best response for the row player against the column player.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_strategy : np.ndarray
        The column player's strategy

    Returns
    -------
    np.ndarray
        The row player's best response
    """

    row_action_values = np.sum(row_matrix * col_strategy[None, ...], axis=1)

    row_br_action = np.argmax(row_action_values)

    row_actions = row_matrix.shape[0]
    
    #This handles one hot encoding
    return np.eye(row_actions)[row_br_action]


def evaluate_row_against_best_response(
    row_matrix: np.ndarray, col_matrix: np.ndarray, row_strategy: np.ndarray
) -> np.float32:
    """Compute the utility of the row player when playing against a best response strategy.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    row_strategy : np.ndarray
        The row player's strategy

    Returns
    -------
    np.float32
        The expected utility of the row player
    """

    column_br = calculate_best_response_against_row(col_matrix, row_strategy)
    player_vals =  evaluate_general_sum(row_matrix, col_matrix, row_strategy, column_br)
    return player_vals[0]


def evaluate_col_against_best_response(
    row_matrix: np.ndarray, col_matrix: np.ndarray, col_strategy: np.ndarray
) -> np.float32:
    """Compute the utility of the column player when playing against a best response strategy.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    col_strategy : np.ndarray
        The column player's strategy

    Returns
    -------
    np.float32
        The expected utility of the column player
    """

    row_br = calculate_best_response_against_col(row_matrix, col_strategy)
    player_vals =  evaluate_general_sum(row_matrix, col_matrix, row_br, col_strategy)

    return player_vals[1]


def find_strictly_dominated_actions(matrix: np.ndarray) -> np.ndarray:
    """Find strictly dominated actions for the given normal-form game.
        finds dominated actions on the first axis. For column player,
        send in the transposed matrix.

    Parameters
    ----------
    matrix : np.ndarray
        A payoff matrix of one of the players

    Returns
    -------
    np.ndarray
        Indices of strictly dominated actions
    """

    #TODO: Perhaps try to vectorize it and get rid of the for loops?
    num_row_actions = matrix.shape[0]
    dominated_actions = [] #[R]
    #Find strictly dominated actions actions
    for a in range(num_row_actions):
        #get all the other actions to check if it
        #is dominated
        #[R-1, C]
        rows_except_a = matrix[np.arange(num_row_actions) != a]
        #[C]
        a_row = matrix[a]
        #Action a is dominated, if some a´ 
        # provides greater value for any action
        # of the oponent.
        if np.any(np.sum(a_row[None, ...] >= rows_except_a, axis=-1) == 0):
            dominated_actions.append(a)
    return np.array(dominated_actions, dtype=np.int32)


def iterated_removal_of_dominated_strategies(
    row_matrix: np.ndarray, col_matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run Iterated Removal of Dominated Strategies.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Four-tuple of reduced row and column payoff matrices, and remaining row and column actions
    """

    r_matrix = row_matrix.copy()
    c_matrix = col_matrix.copy()
    r_actions = np.arange(row_matrix.shape[0])
    c_actions = np.arange(row_matrix.shape[1])
    while True:
        unchanged = True
        #find strictly dominated actions find these actions over the
        # first axis.
        row_dominated = find_strictly_dominated_actions(r_matrix)
        col_dominated = find_strictly_dominated_actions(c_matrix.T)
        if len(row_dominated) > 0:
            unchanged = False
            #breakpoint()
            r_matrix = np.delete(r_matrix, row_dominated, axis=0)
            c_matrix = np.delete(c_matrix, row_dominated, axis=0)
            r_actions = np.delete(r_actions, row_dominated)
        if len(col_dominated) > 0:
            unchanged = False
            r_matrix = np.delete(r_matrix, col_dominated, axis=1)
            c_matrix = np.delete(c_matrix, col_dominated, axis=1)
            c_actions = np.delete(c_actions, col_dominated)
        if unchanged:
            break
    return r_matrix, c_matrix, r_actions, c_actions


def main() -> None:
    pass


if __name__ == '__main__':
    main()
