#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from templates.week01 import calculate_best_response_against_row, evaluate_zero_sum
from scipy.optimize import linprog
from itertools import combinations


def plot_best_response_value_function(row_matrix: np.ndarray, step_size: float) -> None:
    """Plot the best response value function for the row player in a 2xN zero-sum game.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    step_size : float
        The step size for the probability of the first action of the row player
    """
    br_values = []
    row_first_action_probs = []
    num_steps = int(1 / step_size)
    for i in range(num_steps + 1):
        row_first_action_prob = i * step_size
        row_strategy = np.asarray([row_first_action_prob, 1-row_first_action_prob])
        col_br = calculate_best_response_against_row(-row_matrix, row_strategy)
        #This returns us a utility for both players,
        # we want the row player utility
        br_val = evaluate_zero_sum(row_matrix, row_strategy, col_br)[0]
        br_values.append(br_val)
        row_first_action_probs.append(row_first_action_prob)
    row_first_action_probs = np.asarray(row_first_action_probs)
    br_values = np.asarray(br_values)
    fig, ax = plt.subplots()
    ax.plot(row_first_action_probs, br_values)
    ax.set_xlabel("Probability of playing first action of the row player")
    ax.set_ylabel("Utility of the row player against best responding oponent")
    ax.set_label("Best response function of the ")
    plt.show()


def verify_support(
    matrix: np.ndarray, row_support: np.ndarray, col_support: np.ndarray
) -> np.ndarray | None:
    """Construct a system of linear equations to check whether there
    exists a candidate for a Nash equilibrium for the given supports.

    The reference implementation uses `scipy.optimize.linprog`
    with the default solver -- 'highs'. You can find more information at
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html

    This computes the strategy for the column player of the given matrix, hence
    it checks the utility and value for the row player. If you want strategy of the row
    player of the original game, send in transposed matrix of the column player.
    Parameters
    ----------
    matrix : np.ndarray
        A payoff matrix of one of the players
    row_support : np.ndarray
        The row player's support
    col_support : np.ndarray
        The column player's support

    Returns
    -------
    np.ndarray | None
        The opponent's strategy, if it exists, otherwise `None`
    """
    #Just to handle the edge cases when a 1-support is sent 
    # for either player
    one_row = row_support.ndim == 0 or row_support.shape[0] == 1
    one_col = col_support.ndim == 0 or col_support.shape[0] == 1
    if one_row and one_col:
        support_matrix = matrix[row_support, col_support].reshape((1, 1))
    elif (not one_row) and one_col:
        #This is done, because numpy upon receiving
        # multiple integer array indicies combines them into coordianate
        # eg. matrix[[0, 1], [0, 1]] would return only [matrix[0, 0], matrix[1, 1]]
        support_matrix = matrix[row_support[:, None], col_support].reshape((row_support.shape[0], 1))
    elif one_row and (not one_col):
        support_matrix = matrix[row_support, col_support].reshape((1, col_support.shape[0]))
    else:
        support_matrix = matrix[row_support[:, None], col_support]
    # Our variables.
    # x : a vector of size number of column actions 
    # for the distribution
    #  + 1 for the value variable
    #The constraints we want:
    # for each player column player pure action the utility is the
    # same. Can be written as [support_matrix,- ones((R, 1))] @ x = zeros(R).
    # For each row action ar, this evaluates to sum_{ac}(pi(ac) * u_{-i}(ar, ac)) = v{-i}
    # We also cannot forget the constraints that this should be a probability distribution
    # hence [ones(C), 0] @ x = 1 (sums up to one)
    # and lastly x >= zeros(C) (nonegative probabilities, but unbound action value)
    v_multiplicands = -np.ones(support_matrix.shape[0])[..., None]
    dist_multiplicands = np.concatenate([np.ones(support_matrix.shape[1]), np.zeros(1)], axis=0)

    matrix_eq = np.concatenate([support_matrix, v_multiplicands], axis=1)
    A_eq = np.concatenate([matrix_eq, dist_multiplicands[None, ...]], axis=0)
    #A_ub = -np.eye(support_matrix.shape[1] + 1)[:-1]
    #No objective function
    c = np.zeros(support_matrix.shape[1] + 1)
    #b_ub = np.zeros(support_matrix.shape[0])
    #Bounds to handle the fact that first C vars are probabilities
    # and the last one unbound value
    bounds = [(0, 1) for _ in range(support_matrix.shape[1])] + [(None, None)]
    #Handle the probabilities being summed to one
    b_eq = np.concatenate([np.zeros(support_matrix.shape[0]), np.ones(1)])
    # if one_row:
    #     breakpoint()
    res = linprog(c, A_eq = A_eq, b_eq = b_eq, bounds=bounds)
    #If no satisfying solution is found, return
    if not res.success:
        return None
    pi, v = res.x[:-1], res.x[-1]
    return pi


def support_enumeration(
    row_matrix: np.ndarray, col_matrix: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Run the Support Enumeration algorithm and return a list of all Nash equilibria

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        A list of strategy profiles corresponding to found Nash equilibria
    """
    def get_subsets(arr: np.ndarray, start=0):
        # subsets = []
        # for k in range(1, len(arr) + 1):
        #     k_subsets = combinations(arr, k)
        #     for subset in k_subsets:
        #         subsets.append(np.asarray(subset))
        # return subsets
        subsets = []
        if start >= len(arr):
            return []
        for i in range(start, len(arr)):
            subsets.append([arr[i]])
            next_subsets = get_subsets(arr, i + 1)
            for subset in next_subsets:
                subsets.append([arr[i]] + subset)
                #subsets.append(subset)
        return subsets
    row_supports = get_subsets(np.arange(row_matrix.shape[0]))
    expected_rs = (2 ** (row_matrix.shape[0])) - 1
    expected_cs = (2 ** (col_matrix.shape[1])) - 1
    #assert len(row_supports) == expected_rs, f"The number of row supports {len(row_supports)} does not match the expected number of subsets {expected_rs}."
    col_supports = get_subsets(np.arange(col_matrix.shape[1]))
    #assert len(col_supports) == expected_cs, f"The number of col supports {len(col_supports)} does not match the expected number of subsets {expected_cs}." 
    def can_improve_outside_support(matrix: np.ndarray, row_support: np.ndarray, opponent_pi: np.ndarray, current_value):
        """Checks whether a player couldnt have 
        improved his utility by playing an action not supported (with pbt =0) by
        the current strategy.
        Checks for the row player. For column player, send a transposed column
        payout matrix."""
        actions_except_sup = np.delete(np.arange(matrix.shape[0]), row_support, axis=0)
        action_utilities = np.sum(matrix[actions_except_sup] * opponent_pi[None, ...], axis=-1)
        can_improve = np.any(action_utilities > current_value)
        return can_improve
    nash_equilibria = []
    used_supports = set()
    for rs in row_supports:
        for cs in col_supports:
            row_pi = verify_support(col_matrix.T, np.array(cs), np.array(rs))
            if row_pi is None:
                continue
            #Check that all the actions have nonzero 
            # probability, since the LP can return a 
            # solution that has 0 probability on some action     
            row_full_pi = np.zeros(row_matrix.shape[0])
            #print(f"Row full pi shape {row_full_pi.shape}")
            row_full_pi[rs] = row_pi
            col_pi = verify_support(row_matrix, np.array(rs), np.array(cs))
            if col_pi is None:
                continue
            col_full_pi = np.zeros(col_matrix.shape[1])
            col_full_pi[cs] = col_pi
            #We have found a NE candidate. Now we just need to check whether
            # either player could not have gotten a higher utility by choosing an
            # action outside of the support.
            #Here we use the fact, that the value must be same for
            # any pure action of the best response
            row_value = np.sum(row_matrix[rs[0]] * col_full_pi)
            col_value = np.sum(col_matrix[:, cs[0]] * row_full_pi)
            # as those with pbt = 0
            row_support = np.nonzero(row_full_pi > 0)[0]
            col_support = np.nonzero(col_full_pi > 0)[0]
            if can_improve_outside_support(row_matrix, row_support, col_full_pi, row_value):
                continue
            if can_improve_outside_support(col_matrix.T, col_support, row_full_pi, col_value):
                continue
            #This is to make sure we do not repeat the equlibria
            # when another action is added to the support
            # that has no impact on the equilibrium.
            both_supports = (tuple(row_support), tuple(col_support))
            if not both_supports in used_supports:
                used_supports.add(both_supports)
                #Neither player can improve by playing outside the chosen support, 
                # This is a Nash Equilibrium
                nash_equilibria.append((row_full_pi, col_full_pi))
    # print(f"Found nash equilibria: ")
    # for ne in nash_equilibria:
    #     print(f"Equilibrium val {ne}")
    return nash_equilibria


def test_plot_br():
    lecture_matrix = np.array([[-1, 0, -0.8], [1, -1, -0.5]])
    plot_best_response_value_function(lecture_matrix, 0.05)
    plot_best_response_value_function(lecture_matrix, 0.01)
    plot_best_response_value_function(lecture_matrix, 0.001)

def test_verify_support():
    lecture_row_matrix = np.array([[0, 0, -10], [1, -10, -10], [-10, -10, -10]])
    lecture_col_matrix = np.array([[0, 1, -10], [0, -10, -10], [-10, -10, -10]])
    row_support_first = np.array([0, 1])
    col_support_first = np.array([0, 1])
    row_support_second = np.array([0, 2])
    col_support_second = np.array([0, 1])

    col_nash_first = verify_support(lecture_row_matrix, col_support_first, row_support_first)
    assert np.allclose(col_nash_first, np.array([10 / 11, 1 / 11]), atol=1e-5), f"Col nash for the first example should be [10/11, 1/11] not {col_nash_first}"
    row_nash_first = verify_support(lecture_col_matrix.T, row_support_first, col_support_first)
    assert np.allclose(row_nash_first, np.array([10 / 11, 1 / 11]), atol=1e-5), f"Row nash for the first example should be [10/11, 1/11] not {row_nash_first}"

    row_nash_second = verify_support(lecture_col_matrix.T, col_support_second, row_support_second)
    assert np.allclose(row_nash_second, np.array([0, 1]), atol=1e-5) , f"Row nash for the second example should be [0, 1] not {row_nash_second}"
    col_nash_second = verify_support(lecture_row_matrix, row_support_second, col_support_second)
    assert col_nash_second is None, f"Col nash for the second example should not exist, instead got {col_nash_second}"
    #assert np.allclose(col_nash_second, np.array([0, 1]), atol=1e-5) or np.allclose(col_nash_second, np.array([1, 0]), atol=1e-5) , f"Col nash for the second example should be [0, 1] or [1, 0] not {col_nash_second}"

def test_support_enumeration():
    matching_pennies= np.array([[1, -1], [-1, 1]])
    mp_ne = support_enumeration(matching_pennies, -matching_pennies)
    breakpoint()
    rps = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
    rps_ne = support_enumeration(rps, -rps)
    breakpoint()
    battle_of_sexes_row = np.array([[3, 0], [0, 2]])
    battle_of_sexes_col = np.array([[2, 0], [0, 3]])
    bos_ne = support_enumeration(battle_of_sexes_row, battle_of_sexes_col)
    breakpoint()
    prisoner_dillema_row = np.array([[1, 3], [0, 2]])
    prisoner_dillema_col = np.array([[1, 0], [3, 2]])
    pd_ne = support_enumeration(prisoner_dillema_row, prisoner_dillema_col)
    breakpoint()

def main() -> None:
    #test_plot_br()
    #test_verify_support()
    test_support_enumeration()


if __name__ == '__main__':
    main()
