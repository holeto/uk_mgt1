#!/usr/bin/env python3

import numpy as np
from scipy.optimize import linprog


def find_nash_equilibrium(row_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find a Nash equilibrium in a zero-sum normal-form game using linear programming.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A strategy profile that forms a Nash equilibrium
    """
    R = row_matrix.shape[0]
    C = row_matrix.shape[1]
    #Our variables:
    # x, y: a vector of size of 
    # row/column actions (depending who are we solving for)
    # + 1 variable for value
    # (denoting number of row actions as R and column actions as C)
    #For row player we want to solve
    # min x * [zeros(R), -1]
    #s.t. A.t @ p1 - v >= 0 (column player/minimizing player will be best responding)
    # encoded as [-A.t, ones(C)] @ x <= 0
    # p1 >= 0, v unbounded
    # [ones(R), 0] @ x == 1
    c = np.concatenate((np.zeros(R), -np.ones(1)))
    v_multiplicands = np.ones(C)[..., None]
    dist_multiplicands = np.concatenate((np.ones(R), np.zeros(1)))
    bounds = [(0, 1) for _ in range(R)] + [(None, None)]
    A_ub = np.concatenate((-row_matrix.T, v_multiplicands), axis=1)
    res = linprog(c, A_ub = A_ub, b_ub = np.zeros(C), A_eq = dist_multiplicands[None, ...], b_eq = np.ones(1), bounds=bounds)
    assert res.success, "No solution was found for the row player LP. Nash Equilibrium always exists!"
    row_nash, v = res.x[:-1], res.x[-1]
    # For column player
    # min x * [zeros(C), 1]
    # A @ p2 - v <= 0 (row_player/maximizing player will be best responding)
    # [A, -ones(R)] @ x <= 0
    # p2 >= 0
    # [ones(C), 0] @ x == 1
    c = np.concatenate((np.zeros(C), np.ones(1)))
    v_multiplicands = -np.ones(R)[..., None]
    dist_multiplicands = 1 - c
    bounds = [(0, 1) for _ in range(C)] + [(None, None)]
    A_ub = np.concatenate((row_matrix, v_multiplicands), axis=1)
    res = linprog(c, A_ub = A_ub, b_ub = np.zeros(R), A_eq = dist_multiplicands[None, ...], b_eq = np.ones(1), bounds=bounds)
    assert res.success, "No solution was found for the column player LP. Nash Equilibrium always exists!"
    col_nash, v = res.x[:-1], res.x[-1]
    return row_nash, col_nash



def find_correlated_equilibrium(row_matrix: np.ndarray, col_matrix: np.ndarray) -> np.ndarray:
    """Find a correlated equilibrium in a normal-form game using linear programming.

    While the cost vector could be selected to optimize a particular objective, such as
    maximizing the sum of players’ utilities, the reference solution sets it to the zero
    vector to ensure reproducibility during testing.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix

    Returns
    -------
    np.ndarray
        A distribution over joint actions that forms a correlated equilibrium
    """
    R = row_matrix.shape[0]
    C = col_matrix.shape[1]
    # Our variables:
    # x: R * C variables, one for 
    # the probability of each joint action.
    # Here, for simplicity we assume it can be indexed as
    # x[ar, ac] (not how it is actually implemented)
    # We are solving
    # min 0
    # s.t.
    # row_matrix[ar, :] @ x[ar, :] >= row_matrix(a´r, :) @ x[ar, :] for each ar, a´r
    # or, alternatively (row_matrix(a´r, :) - row_matrix[ar, :]) @ x[ar, :] <= 0 for each ar, a´r
    # similarly for column player
    # (col_matrix[:, a´c] - col_matrix)[:, ac] @ x[:, ac] <= 0
    # x >= 0
    # sum(x) == 1
    N = R * C
    A_ub = []
    #add the row action constraints
    for r1 in range(R):
        for r2 in range(R):
            if r2 == r1:
                continue
            one_constraint = np.zeros(N)
            one_constraint[(r1 * C): ((r1 + 1) * C)] = -row_matrix[r1, :] + row_matrix[r2, :]
            A_ub.append(one_constraint)
    
    row_range = np.arange(0, R * C, C)
    #add the column action contraints
    for c1 in range(C):
        for c2 in range(C):
            if c1 == c2:
                continue
            one_constraint = np.zeros(N)
            one_constraint[row_range + c1] = -col_matrix[:, c1] + col_matrix[:, c2]
            A_ub.append(one_constraint)

    A_ub = np.stack(A_ub, axis=0)
    b_ub = np.zeros(A_ub.shape[0])
    A_eq = np.ones((1, N))
    b_eq = np.ones(1)
    bounds = [(0, 1) for i in range(N)]

    #Choosing the objective function allows us
    # to find a correlated equilibrium with a particular property.
    # Here, we are intersted in finding any one so we dont
    # have any
    c = np.zeros(N)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq = A_eq, b_eq=b_eq, bounds=bounds)

    assert res.success, "No solution was found for the LP. Correlated equilibrium should always exist!"
    correlated_equilibrium = np.reshape(res.x, (R, C))

    return correlated_equilibrium


def main() -> None:
    pass


if __name__ == '__main__':
    main()
