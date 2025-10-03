#!/usr/bin/env python3

import numpy as np

def evaluate_pair(
    row_strategy: np.ndarray, col_strategy: np.ndarray,
    row_matrix: np.ndarray, col_matrix: np.ndarray
) -> np.ndarray:
    """ Compute the expected utility of each player in a general-sum game.

    Parameters
    ----------
    row_strategy : np.ndarray
        The row player's strategy
    col_strategy : np.ndarray
        The column player's strategy
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix

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


def evaluate(
    row_strategy: np.ndarray, col_strategy: np.ndarray, matrix: np.ndarray
) -> np.ndarray:
    """ Compute the expected utility of each player in a zero-sum game.

    Parameters
    ----------
    row_strategy : np.ndarray
        The row player's strategy
    col_strategy : np.ndarray
        The column player's strategy
    matrix : np.ndarray
        The row player's payoff matrix

    Returns
    -------
    np.ndarray
        A vector of expected utilities of the players
    """
    p1_util = np.sum(matrix * row_strategy[..., None] * col_strategy[None, ...])
    return np.asarray([p1_util, -p1_util])


def best_response_strategy_against_row(
    row_strategy: np.ndarray, col_matrix: np.ndarray
) -> np.ndarray:
    """ Compute a pure best response strategy of the column player against the row player.

    Parameters
    ----------
    row_strategy : np.ndarray
        The row player's strategy
    col_matrix : np.ndarray
        The column player's payoff matrix

    Returns
    -------
    np.ndarray
        The column player's strategy
    """
    #action values for col actions are weighted average over
    # row actions
    col_action_values = np.sum(col_matrix * row_strategy[..., None], axis=0)

    col_br_action = np.argmax(col_action_values)

    return col_br_action


def best_response_strategy_against_col(
    col_strategy: np.ndarray, row_matrix: np.ndarray
) -> np.ndarray:
    """ Compute a pure best response strategy of the row player against the column player.

    Parameters
    ----------
    col_strategy : np.ndarray
        The column player's strategy
    row_matrix : np.ndarray
        The row player's payoff matrix

    Returns
    -------
    np.ndarray
        The row player's strategy
    """
    
    row_action_values = np.sum(row_matrix * col_strategy[None, ...], axis=1)

    row_br_action = np.argmax(row_action_values)

    return row_br_action


def evaluate_row_against_best_response(
    row_strategy: np.ndarray, col_matrix: np.ndarray
) -> np.ndarray:
    """ Compute the utilities when the row player plays against a best response strategy.

    Note that this function works only for zero-sum games

    Parameters
    ----------
    row_strategy : np.ndarray
        The row player's strategy
    col_matrix : np.ndarray
        The column player's payoff matrix

    Returns
    -------
    np.ndarray
        A vector of expected utilities of the players
    """

    column_br = best_response_strategy_against_row(row_strategy, col_matrix)
    num_actions = row_strategy.shape[0]
    column_br_oh = np.eye(num_actions)[column_br]
    #This is one option how to get it, another
    # would be evaluate_both(row_stategy, column_br_oh, -col_matrix, col_matrix)
    player_vals = evaluate(row_strategy, column_br_oh, -col_matrix)
    return player_vals


def evaluate_col_against_best_response(
    col_strategy: np.ndarray, row_matrix: np.ndarray
) -> np.ndarray:
    """ Compute the utilities when the column player plays against a best response strategy.

    Note that this function works only for zero-sum games

    Parameters
    ----------
    col_strategy : np.ndarray
        The column player's strategy
    row_matrix : np.ndarray
        The row player's payoff matrix

    Returns
    -------
    np.ndarray
        A vector of expected utilities of the players
    """
    row_br = best_response_strategy_against_col(col_strategy, row_matrix)
    num_actions = col_strategy.shape[0]
    #This handles one-hot encoding
    row_br_oh = np.eye(num_actions)[row_br]
    #This is one option how to get it, another
    # would be evaluate_both(row_stategy, column_br_oh, row_matrix, -row_matrix)
    player_vals = evaluate(row_br_oh, col_strategy, row_matrix)
    return player_vals


def dominated_actions(
    row_matrix: np.ndarray, col_matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """ Find strictly dominated actions for each player.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A sequence of indices of dominated actions for each player.
    """

    #TODO: Perhaps try to vectorize it and get rid of the for loops?
    num_row_actions = row_matrix.shape[0]
    num_column_actions = col_matrix.shape[1]
    row_dominated_actions = [] #[R]
    col_dominated_actions = [] #[C]
    #Find dominated row actions
    for a_r in range(num_row_actions):
        #get all the other actions to check if it
        #is dominated
        #[R-1, C]
        rows_except_a = row_matrix[np.arange(num_row_actions) != a_r]
        #[C]
        a_row = row_matrix[a_r]
        #Action a is dominated, if some a´ 
        # provides greater value for any action
        # of the oponent.
        if np.any(np.sum(a_row[None, ...] >= rows_except_a, axis=-1) == 0):
            row_dominated_actions.append(a_r)
       
    #Find dominated column actions
    for a_c in range(num_column_actions):
        #get all the other actions to check if it
        #is dominated
        #[R, C-1]
        cols_except_a = col_matrix[:, np.arange(num_column_actions) != a_c]
        #[R]
        a_col = col_matrix[:, a_c]
        #Action a is not dominated, if it provides a greater 
        # value than some a´ for some action of the oponent
        if np.any(np.sum(a_col[..., None] >= cols_except_a, axis=0) == 0):
            col_dominated_actions.append(a_c)
    #breakpoint()
    return np.array(row_dominated_actions), np.array(col_dominated_actions)


def iterated_removal_of_dominated_actions(
    row_matrix: np.ndarray, col_matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Run the Iterated Removal of Dominated Actions algorithm.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A pair of reduced payoff matrices and a pair of remaining actions for each player
    """
    r_matrix = row_matrix.copy()
    c_matrix = col_matrix.copy()
    r_action_mask = np.ones(row_matrix.shape[0], dtype=np.bool)
    c_actions_mask = np.ones(row_matrix.shape[1], dtype=np.bool)
    while True:
        unchanged = True
        row_dominated, col_dominated = dominated_actions(r_matrix, c_matrix)
        for row_da in row_dominated:
            unchanged = False
            r_action_mask[row_da] = False
            r_matrix = np.delete(r_matrix, row_da, axis=0)
            c_matrix = np.delete(c_matrix, row_da, axis=0)
        for col_da in col_dominated:
            unchanged = False
            c_actions_mask[col_da] = False
            r_matrix = np.delete(r_matrix, col_da, axis=1)
            c_matrix = np.delete(c_matrix, col_da, axis=1)
        if unchanged:
            break
    r_actions = np.arange(row_matrix.shape[0])[r_action_mask]
    c_actions = np.arange(row_matrix.shape[1])[c_actions_mask]
    return r_matrix, c_matrix, r_actions, c_actions


def main() -> None:
    pass
   
if __name__ == '__main__':
    main()
