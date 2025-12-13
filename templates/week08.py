#!/usr/bin/env python3

import numpy as np

from scipy.optimize import linprog
from templates.week07 import *
from templates.week04 import find_nash_equilibrium
from templates.week01 import evaluate_zero_sum

i32 = np.int32

def cartesian_product(*arrays):
    """Implementation of cartesian product of 
    N 1D arrays. Taken from https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points"""
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def collect_infoset_actions(efg: EFG) ->tuple[list[np.ndarray], list[np.ndarray]]:
    """Collect all distinct actions (eg. differentiated by infosets) for both players 


    Args:
        efg (EFG): The extensive form game.

    Returns:
        tuple[list[np.ndarray], list[np.ndarray]: The lists of all distinct actions for both players.
        Each list is sliced per depth in the game tree, same as behavioral strategies.
        Each action is represented as an integer i, 
        where i // efg.num_actions is the index of the infoset in efg.depth_infoset_map[depth]
        , that the action was taken in, and
        i % efg.num_actions the index of the action itself.
    """
    assert efg.num_players == 2 , f"The collection of infoset actions is implemented only for 2 player efgs so far. Got {efg.num_players} players"
    pl_actions = ([], [])
    for d in range(efg.max_depth):
        for pl in range(efg.num_players):
            #At each depth, the first infoset
            # of each player is the invalid one.
            # Skip it.
            iset_legal = efg.depth_iset_legal[d][pl]
            num_isets = iset_legal.shape[0]
            if num_isets == 1:
                #Append just one dummy action here 
                pl_actions[pl].append(np.zeros(1, dtype=i32))
                continue
            #Remove the first dummy iset
            iset_legal = iset_legal[1:]
            iset_actions = np.arange(iset_legal.size, dtype=i32)
            #These are the legal actions we can take
            iset_actions = iset_actions[iset_legal.ravel().astype(bool)]
            # Do not forget to add num_actions, to compensate for the dummy infoset offset.
            pl_actions[pl].append(iset_actions + efg.num_actions)
    return pl_actions

def infoset_actions_to_pure_strategies(infoset_actions: list[np.ndarray], num_actions: int):
    """Convert a per depth sliced actions for each infoset to all pure strategies in the game
    This will be for a single player infoset_actions.

    Args:
        infoset_actions (list[np.ndarray]): Per depth array of infoset action
        identificators. As produced by collect_infoset_actions.

    Returns:
        An array of pure strategies of shape (num_pure_strategies, num player isets in the game)
        and a list of remaping keys, where at index i is a tuple of (depth, infoset_index_at depth)
        of lenght num player isets in the game.
    """
    #The easiest way to convert
    # to per iset grouping is likely using a dictionary
    actions_by_infoset = {}

    for depth, actions_at_depth in enumerate(infoset_actions):
        for iset_action_id in actions_at_depth:
            #This is an invalid infoset we do not play here
            if iset_action_id == 0:
                continue
            #Decode the indices of infoset and action
            iset_idx = iset_action_id // num_actions
            action_id = iset_action_id % num_actions
            
            #Assuming this will identify a unique infoset
            # (here it will due to perfect recall)
            key = (depth, iset_idx)
            if key not in actions_by_infoset:
                actions_by_infoset[key] = []
            actions_by_infoset[key].append(action_id)

    #To make sure that it is ordered by (depth, infoset_index)
    depth_index_keys = sorted(actions_by_infoset.keys())
    
    action_lists = [np.array(actions_by_infoset[k]) for k in depth_index_keys]

    pure_strategies = cartesian_product(*action_lists)

    return pure_strategies, depth_index_keys

def pure_strategies_to_behavioral(efg: EFG, player_pure_strategies: tuple[np.ndarray], 
                                  player_maps: tuple[list]) ->list[list[np.ndarray]]:
    """Take a pure strategies for each player and convert them into the behavioral form

    Args:
        efg (EFG): The extensive form game
        player_pure_strategies (tuple[np.ndarray]): A pure strategy for each
        player. Where each pure strategy is an array defining action for each
        infoset.
        player_maps (tuple[list]): A list for each player, where at
        index is are indices (depth, depth_index), to define the remaping of infoset
        i to the per depth form.

    Returns:
        list[list[np.ndarray]]: A per depth per player behavioral strategies for 
        each infoset at the depth. 
    """
    behaviorals = [[np.zeros(pl.shape)for pl in d] for d in efg.depth_iset_legal]
    for pl, pl_data in enumerate(zip(player_pure_strategies, player_maps)):
        strat, pl_map = pl_data
        for i, a in enumerate(strat):
            depth, idx = pl_map[i]
            behaviorals[depth][pl][idx][a] = 1.0
    return behaviorals


def convert_to_normal_form(efg: EFG) -> tuple[np.ndarray, np.ndarray]:
    """Convert an extensive-form game into an equivalent normal-form representation.
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A pair of payoff matrices for the two players in the resulting normal-form game.
    """
    assert efg.num_players == 2 , f"The conversion to normal form is implemented only for 2 player efgs so far. Got {efg.num_players} players"
    p1_infoset_actions, p2_infoset_actions = collect_infoset_actions(efg)
    p1_pure_strategies, p1_map = infoset_actions_to_pure_strategies(p1_infoset_actions, efg.num_actions)
    p2_pure_strategies, p2_map = infoset_actions_to_pure_strategies(p2_infoset_actions, efg.num_actions)
    R = p1_pure_strategies.shape[0]
    C = p2_pure_strategies.shape[0]
    row_matrix = np.zeros((R, C))
    col_matrix = np.zeros((R, C))
    for i, p1_strat in enumerate(p1_pure_strategies):
        for j, p2_strat in enumerate(p2_pure_strategies):
            behaviorals = pure_strategies_to_behavioral(efg, (p1_strat, p2_strat), (p1_map, p2_map))
            expected_utils = evaluate(efg, behaviorals)
            row_matrix[i, j] = expected_utils[0]
            col_matrix[i, j] = expected_utils[1]
    return row_matrix, col_matrix


def pure_sequences_to_realization(seq_indices: Iterable[int], seq_maps: Iterable[dict]) ->list[np.ndarray]:
    """Convert a sequence identified by its index
    into its pure realization plan.

    Args:
        seq_idx (list[int]): Index into the sequence map
        defining the sequence. One for each player
        seq_map (dict): A dict indexed by sequence indices.
        For each sequence contains (parent_sequence_index, depth, iset_idx, action_idx).
        One dict for each player
    """
    realizations = []
    for seq_idx, seq_map in zip(seq_indices, seq_maps):
        #Do not forget to add 1 for the empty sequence.
        # that one is not in the sequence map
        S = len(seq_map) + 1
        #The empty sequence always has realization plan of 1
        realization = np.eye(S)[0]
        #0 is the index reserved for the empty sequence
        while seq_idx > 0:
            realization[seq_idx] = 1
            #Backtrack to the parent sequence
            seq_idx = seq_map[seq_idx][0]
        realizations.append(realization)
    return realizations

def collect_sequences(efg: EFG):
    """
    Collects sequences for both players.
    
    Returns:
        valid_sequences: A set of tuples (p1_seq, p2_seq), defining sequences
        which lead to a leaf node.
        seq_maps: A tuple (p1_map, p2_map). 
                         Each is a dictionary: seq_id -> (parent_seq_id, depth_idx, iset_idx, action_idx)
                         Intended to allow easy conversion to behavioral strategies.
        seq_successors: A tuple (p1_succ, p2_succ).
                        Each is a dictionary: (depth, iset_id) -> (parent_sequence_id, list of sequence ids of successors).
                        Intended to allow easy construction of the realization plan constraint matrix.
    """
    

   
    
    # Track the next available Sequence ID for each player. Start at 1.
    # (0 is reserved for the root/empty sequence)
    sequence_counters = [1 for pl in range(efg.num_players)]

    
    seq_maps = [{}, {}]
    #Make this a set, since chance nodes could lead to sequence duplication
    # and unnecessary computation.
    valid_sequences = set()
    seq_successors = [{}, {}]
    #Per history sequences
    current_seqs = np.zeros((efg.num_players, 1), dtype=i32)

    for d in range(efg.max_depth):
        cur_iset = efg.depth_isets[d]
        cur_iset_legal = efg.depth_iset_legal[d]
        cur_player = efg.depth_history_player[d]

        next_seqs = np.zeros((efg.num_players, *efg.depth_history_legal[d].shape), dtype=i32)
        non_terminal = np.logical_and((efg.depth_history_next_history[d] >= 0), efg.depth_history_legal[d].astype(bool)).ravel()         
        terminal = np.logical_and(efg.depth_history_next_history[d] == -1,efg.depth_history_legal[d].astype(bool)).ravel()

        for pl in range(efg.num_players):
            player_mask = cur_player == pl
            #These are for when the player does not act in the given
            # history. Used to propagate the same sequence downward.
            repeat_seqs = np.repeat(current_seqs[pl, ..., None], efg.num_actions, axis=-1)
            if np.sum(player_mask) == 0:
                #If the player does not act at this depth
                #just propagate his sequences as identical
                next_seqs[pl] = repeat_seqs
                continue
            player_isets = cur_iset[player_mask]
            #Due to perfect recall, we can just propagate the sequences
            # over isets, since they will be the same for any history under
            # the given infoset.

            #Ignore the first invalid iset.
            player_iset_legal = cur_iset_legal[pl][1:]
                
            
            #Our infoset differentiated actions are
            # what we will be adding to the sequence
            #Also do not forget to add the id 
            # to differentiate from the action at previous depths
            iset_actions = ((np.cumsum(player_iset_legal.ravel()) * player_iset_legal.ravel()) - 1).reshape(player_iset_legal.shape)
            
            next_iset_seqs = np.where(iset_actions >= 0, iset_actions + sequence_counters[pl], iset_actions)
            #First copy the sequences for the histories
            # where the player does not act
            next_seqs[pl] = repeat_seqs
            #Then get the new sequences for the state
            # Where the player does act
            # Do not forget the -1 to compensate for
            # The invalid iset offset.
            next_seqs[pl, player_mask] = next_iset_seqs[player_isets - 1]
            #Get the first occurence of iset
            # in the histories, to retreive the parent sequence.
            # Due to perfect recall, we can just take the first occurence,
            # all others will be the same.
            _, p_indices = np.unique(player_isets, return_index=True)
            pl_seqs = current_seqs[pl, player_mask]
            parent_sequences = pl_seqs[p_indices]
            
            for i, iset_seq in enumerate(next_iset_seqs):
                p_seq = int(parent_sequences[i])
                #Store per iset the parent sequence
                # and successor sequences
                iset_key = (d, i) 
                seq_successors[pl][iset_key] = (p_seq, [s for s in iset_seq if s >= 0])
                # + 1 to compensate for the invalid iset offset
                iset_id = i + 1
                for a, seq in enumerate(iset_seq):
                    #Illegal sequence
                    if seq < 0:
                        continue
                    # Store: Parent sequence index, depth, infoset_index and action index
                    # for the remaping to behavioral strategies.
                    seq_maps[pl][int(seq)] = (p_seq, d, iset_id, a)

            num_new = np.sum(player_iset_legal, dtype=i32)
            sequence_counters[pl] += num_new
        #Do not further expand the terminal or illegal sequences
        next_seqs = np.reshape(next_seqs, (efg.num_players, -1))
        #Store the sequences that reach a leaf node
        terminal_seqs = next_seqs[:, terminal].astype(i32)
        num_terminal = terminal.sum()
        for i in range(num_terminal):
            valid_seqs = tuple(terminal_seqs[:, i])
            valid_sequences.add(valid_seqs)
        current_seqs = next_seqs[:, non_terminal].astype(i32)
    return valid_sequences, seq_maps, seq_successors



def convert_to_sequence_form(efg:EFG) -> tuple[tuple[np.ndarray], tuple[np.ndarray]]:
    """Convert an extensive-form game into its sequence-form representation.

    The sequence-form representation consists of:
        - The sequence-form payoff matrices for both players
        - The realization-plan constraint matrices and vectors for both players
    Returns
    -------
    tuple[tuple[np.ndarray], tuple[np.ndarray], tuple(dict)]
        A tuple containing the sequence-form payoff matrices and realization-plan constraints.
        Ordered as payoff matrices tuple, constraint matrices tuple.
        And also a tuple  of dictionaries of sequence mappings for each player, that contains for
        each sequence id a tuple (parent_sequence_id, depth, iset_index, action_index)
    """

    assert efg.num_players == 2, "The sequence form conversion is implemented only for 2 player games."
    valid_sequences, seq_maps, seq_succesors = collect_sequences(efg)
    #The sequence maps do not contain the first
    # empty sequence. Do not forget to add it.
    S = [len(m) + 1 for m in seq_maps]

    
    #Construct the realization plan constraint matrices
    constraint_matrices = []
    for pl in range(efg.num_players):
        #Do not forget to add the first row
        # that will ensure that realization of the
        # empty sequence is equal to 1.
        first_row = np.eye(S[pl], dtype=i32)[0] 
        mat = [first_row]
        for iset_data in seq_succesors[pl].values():
            row = np.zeros(S[pl], dtype=i32)
            parent, succ = iset_data
            row[parent] = -1
            row[np.array(succ)] = 1
            mat.append(row)
        mat = np.stack(mat, axis=0)
        constraint_matrices.append(mat)
    #Construct the payoff matrices.
    #This part uses the two player assumption
    # to avoid some tensor complications
    A = np.zeros((S[0], S[1]))
    B = np.zeros_like(A)
    for seq_tup in valid_sequences:
        #We first create a pure realizations plan for these sequences
        seq_realizations = pure_sequences_to_realization(seq_tup, seq_maps)
        #The conversion to behaviorals allows us to use
        # expected utility computation to get the utility
        # if the sequences would not lead us to terminal history
        # we automatically get zero
        # without need for any additional checking, due to it being multiplied by 0 reach.
        behaviorals = convert_realization_plan_to_behavioural_strategy(efg, seq_realizations,
                                                                        seq_maps)

        seq_utilities = evaluate(efg, behaviorals)
        A[*seq_tup] = seq_utilities[0]
        B[*seq_tup] = seq_utilities[1]
            
    return (A, B), tuple(constraint_matrices), tuple(seq_maps)

    



def find_nash_equilibrium_sequence_form(efg: EFG) -> tuple[float, list[list[np.ndarray]]]:
    """Find a Nash equilibrium in a zero-sum extensive-form game using Sequence-form LP.

    This function is expected to received an extensive-form game as input
    and convert it to its sequence-form using `convert_to_sequence_form`.

    Returns
    -------
    tuple[float, list[list[np.ndarray]]]
        Game value from player 1 point of view followed by
        the per depth per player iset behavioral strategies induced
        by the found realization plans.
    """
    payoffs, constr, seq_maps = convert_to_sequence_form(efg)
    A, B = payoffs
    E, F = constr
    assert not np.any(np.abs(A + B) >= 1e-8), f"SQFLP only finds Nash equilibrium for zero sum games. Matrices {A} \n and {B} \n do not form a zero sum game!"
    S1, S2 = A.shape
    I1, I2 = E.shape[0], F.shape[0]
    e = np.eye(I1)[0]
    f = np.eye(I2)[0]
    #The variables are utility vectors ui
    # of shape Ii and realization plans ri of shape Si
    
    #For first player we want to solve
    # min u1.t @ e
    # F @ r2 = f
    # -E.t @ u1 <= -A @ r2 or [-E.t, A] @ [u1, r2] <= 0
    # r2 >= 0
    # And gives us the value of the game
    # in u1[0] and the minmax strategy of opponent defined by r2

    #We just need padding because linprog requires a single variable vector
    c = np.concatenate([e, np.zeros(S2)])
    A_eq = np.concatenate([np.zeros((I2, I1)), F], axis=-1)
    A_ub = np.concatenate([-E.T, A], axis=-1)
    bounds = [(-np.inf, np.inf) for _ in range(I1)] + [(0, 1) for _ in range(S2)]
    res = linprog(c, A_ub = A_ub, b_ub = np.zeros(S1), A_eq= A_eq, b_eq = f, bounds=bounds)
    assert res.success, "No Nash realization plan was found for player 2, but it should always exist!"
    p1_val = res.x[0]
    r2 = res.x[I1:]

    #For second player we want to solve
    # max u2.t @ f or min u2.t @ -f
    # E @ r1  = e
    # F.t @ u2 <= A.t @ r1 or [F.t, -A.t] @ [u2, r1] <= 0
    # r1 >= 0
    # And we get the game value in u2[0].
    # and maximin strategy of player 1 defined by r1
    c = np.concatenate([-f, np.zeros(S1)])
    A_eq = np.concatenate([np.zeros((I1, I2)), E], axis=-1)
    A_ub = np.concatenate([F.T, -A.T], axis=-1)
    bounds = [(-np.inf, np.inf) for _ in range(I2)] + [(0, 1) for _ in range(S1)]
    res = linprog(c, A_ub = A_ub, b_ub = np.zeros(S2), A_eq= A_eq, b_eq = f, bounds=bounds)
    assert res.success, "No Nash realization plan was found for player 1, but it should always exist!"
    p2_val = res.x[0]
    assert np.abs(p1_val - p2_val) < 1e-8, f"The found game values by the player 2 solving LP {p1_val} and player 1 solving LP {p2_val} do not match!" 
    r1 = res.x[I2:]
    behaviorals = convert_realization_plan_to_behavioural_strategy(efg, (r1, r2), seq_maps)

    return p1_val, behaviorals


def convert_realization_plan_to_behavioural_strategy(efg: EFG, realizations: Iterable[np.ndarray], seq_maps: Iterable[dict]):
    """Convert a realization plan to a behavioural strategy.

    Args:
        efg (EFG): Extensive form game
        realizations (Iterable[np.ndarray]): The realization plans of all players
        seq_maps (Iterable[dict]): The sequence maps for each player, where
        for each sequence index they contain (parent_sequence_index, depth, infoset_index, action_index)
    """

    behaviorals = [[np.zeros(pl.shape)for pl in d] for d in efg.depth_iset_legal]
    for pl, pl_r in enumerate(realizations):
        #Skip the realization plan of the empty sequence
        for i, r in enumerate(pl_r[1:]):
            #Skip the 0 probability realizations
            if r < 1e-8:
                continue
            parent, depth, iset_idx, action = seq_maps[pl][i + 1]
            #The probability of taking that action is
            parent_prob = pl_r[parent]
            if r > 1e-8 and parent_prob < 1e-8:
                raise ValueError(f"The child sequence realization {r} is nonzero, but the parent sequence realization {parent_prob} is close to zero. The realization plan is not well formed!")
            # realization prob of the sequence / realization prob of the parent
            behaviorals[depth][pl][iset_idx, action] = r / parent_prob
    return behaviorals

def solve_nfg(efg: EFG):
    """Convert the 2p0s extensive form game to normal form game,
    and solve it with LP"""
    assert efg.num_players == 2, f"Conversion to NFG and solving with LP is only for 2p0s games. Got number of players {efg.num_players}"
    row_matrix, col_matrix = convert_to_normal_form(efg)
    #check if it is zero sum
    if np.any((row_matrix + col_matrix) ** 2 >= 1e-8):
        assert False, f"Matrices {row_matrix} \n and {col_matrix} \n do not form a zero sum game!"
    p1_strategy, p2_strategy = find_nash_equilibrium(row_matrix)
    game_value = evaluate_zero_sum(row_matrix, p1_strategy, p2_strategy)
    #print(f"Found a Nash equilibrium of the game with game value {game_value}")
    return game_value[0]

def solve_sequence_form(efg: EFG):
    game_value, behaviorals = find_nash_equilibrium_sequence_form(efg)
    evaluated_game_value = evaluate(efg, behaviorals)[0]
    assert np.abs(game_value - evaluated_game_value) <  1e-8, f"Game value found by the SQFLP {game_value} and game value found by the returned behaviorals {evaluated_game_value} do not match!"
    return game_value


def main() -> None:
    game = KuhnPoker()
    efg = traverse_tree(game)
    seq_game_value = solve_sequence_form(efg)
    nf_game_value = solve_nfg(efg)
    assert np.all(np.isclose(seq_game_value, -1 / 18, atol=1e-8)), f"The value of the game for Kuhn poker should be close to -1/18. Instead got {seq_game_value} (difference {np.abs(seq_game_value + (1 / 18))})"
    assert np.all(np.isclose(nf_game_value, -1 / 18, atol=1e-8)), f"The value of the game for Kuhn poker should be close to -1/18. Instead got {nf_game_value} (difference {np.abs(nf_game_value + (1 / 18))})"


if __name__ == '__main__':
    main()
