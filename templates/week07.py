#!/usr/bin/env python3

"""
Extensive-form games assignments.

Starting this week, the templates will no longer contain exact function
signatures and there will not be any automated tests like we had for the
normal-form games assignments. Instead, we will provide sample outputs
produced by the reference implementations which you can use to verify
your solutions. The reason for this change is that there are many valid
ways to represent game trees (e.g. flat array-based vs. pointer-based),
information sets and strategies in extensive-form games. Figuring out
the most suitable representations is an important part of assignments
in this block. Unfortunately, this freedom makes automated testing
pretty much impossible.
"""

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os

from typing import Iterable
from dataclasses import dataclass
from copy import deepcopy

from pgx import Env
from templates.kuhn_poker import State, KuhnPoker

i32 = np.int32

@dataclass
class EFG():
    #Denoting H(D) as the amount of histories in the given depth
    # S(D) the amount of infosets at the given depth
    # A the maximal amount of actions per infoset
    # Pl the amount of players in the game
    # D the maximal depth reachable in the game tree

    depth_isets: list[np.ndarray] #iset indices per history [D, H(D)]
    depth_history_legal: list[np.ndarray] # legal actions per history [D, H(D), A]
    depth_history_actions: list[np.ndarray] # action indices per history. Actions are differentianted per isets [D, H(D), A]
    depth_history_utility: list[np.ndarray] # utility for each player after the given action is played [D, H(D), A, Pl]
    depth_history_next_history: list[np.ndarray] # The index of the next history that the action will lead to. -1 if terminal [D, H(D), A]
    depth_history_player: list[np.ndarray] #player index for each history [D, H(D)]
    depth_history_is_chance: list[np.ndarray] #A flag whether the state is chance [D, H(D)]
    depth_history_chance_probabilities: list[np.ndarray] #Chance outcome probabilities [D, H(D), A]

    depth_iset_map: list[list[np.ndarray]] #Iset maps for all players at the given depth, where iset indexed as i is stored at index i [D, Pl, S(D)]
    depth_iset_legal: list[list[np.ndarray]] # Legal actions for each iset for each player at the given depth [D, Pl, S(D), A]

    num_players: int
    num_actions: int
    max_depth: int

def _construct_iset_map(num_players: int, num_actions: int,
                        history_isets: np.ndarray, history_legal: np.ndarray, history_player: np.ndarray, history_is_chance: np.ndarray):
    #Convert to numpy, it is faster to iterate over it
    # than JAX arrays.
    history_isets = np.asarray(history_isets)
    boolean_obs = history_isets.dtype == bool
    history_legal = np.asarray(history_legal)
    history_player = np.asarray(history_player)
    history_is_chance = np.asarray(history_is_chance)

    invalid_iset = np.zeros_like(history_isets[0])
    invalid_legal = np.ones_like(history_legal[0])

    iset_map = [[invalid_iset] for pl in range(num_players)]
    legal_map = [[invalid_legal] for pl in range(num_players)]
    history_iset_indices = np.full(history_isets.shape[0], -1, dtype=i32)
    history_actions = np.full((history_isets.shape[0], num_actions), -1, dtype=i32)
    actions = np.arange(num_actions)
    def get_idx(map, search_iset):
        for i, iset in enumerate(map):
            isets_equal = (boolean_obs and np.all(search_iset == iset)) or ((not boolean_obs) and np.sum((search_iset - iset) ** 2) <= 1e-10)
            if isets_equal:
                return i
        return -1
    i = 0
    for iset, legal, pl, chance in zip(history_isets, history_legal, history_player, history_is_chance):
        if chance:
            iset_idx = 0
        else:
            iset_idx = get_idx(iset_map[pl], iset)
            if iset_idx < 0:
                iset_idx = len(iset_map[pl])
                iset_map[pl].append(iset)
                legal_map[pl].append(legal)

        history_iset_indices[i] = iset_idx
        history_actions[i] = actions + iset_idx * num_actions
        i += 1
    iset_map = [np.asarray(m) for m in iset_map]
    legal_map = [np.asarray(m) for m in legal_map]
    return iset_map, legal_map, history_iset_indices, history_actions
        

def traverse_tree(game: Env):
    """Build a full extensive-form game tree for a given game."""

    #Can just use a dummy key for full tree traversal
    init_state = game.init(jax.random.key(0))

    #Expand the dimensions we count that it will always be present
    init_state = jax.tree.map(lambda x: x[None, ...], init_state)

    #The outer vmap is over H(D), the inner over the actions, so we do not need to vmap over the state in the inner vmap
    vectorized_apply_action = jax.vmap(jax.vmap(game.step, in_axes=(None, 0), out_axes=(0)), in_axes=(0, 0), out_axes=0)

    depth_isets = []
    depth_history_legal = []
    depth_history_actions = []
    depth_history_utility = []
    depth_history_next_history = []
    depth_history_player = []
    depth_history_is_chance = []
    depth_history_chance_probabilities = []

    depth_iset_map = []
    depth_iset_legal = []

    def construct_tree(state: State, depth=0):
        iset_map, iset_legal, isets, action_indices = _construct_iset_map(game.num_players, game.num_actions,
                                                                  state.observation, state.legal_action_mask, state.current_player, state.is_chance_node)
        actions = jnp.tile(jnp.arange(game.num_actions), (isets.shape[0], 1))
        next_states = vectorized_apply_action(state, actions)

        non_terminal = (~next_states.terminated * ~next_states.truncated * state.legal_action_mask)
        non_terminal = np.asarray(non_terminal)

        next_histories = (np.cumsum(non_terminal, dtype=i32) * non_terminal.ravel()) - 1

        next_histories = next_histories.reshape(non_terminal.shape)

        next_rewards = next_states.rewards * state.legal_action_mask[..., None]
        
        depth_isets.append(np.asarray(isets, isets.dtype))
        depth_history_legal.append(np.asarray(state.legal_action_mask, state.legal_action_mask.dtype))
        depth_history_actions.append(np.asarray(action_indices, action_indices.dtype))
        depth_history_utility.append(np.asarray(next_rewards, next_rewards.dtype))
        depth_history_next_history.append(np.asarray(next_histories, next_histories.dtype))
        depth_history_player.append(np.asarray(state.current_player, state.current_player.dtype))
        depth_history_is_chance.append(np.asarray(state.is_chance_node, state.is_chance_node.dtype))
        depth_history_chance_probabilities.append(np.asarray(state.chance_strategy, state.chance_strategy.dtype))

        depth_iset_map.append(iset_map)
        depth_iset_legal.append(iset_legal)

        if np.sum(non_terminal) == 0:
            return
        nonzeros = np.flatnonzero(non_terminal)
        #flatten the first two dimensions of next_states into one
        next_states = jax.tree.map(lambda x: jnp.reshape(x, (-1, *x.shape[2:])), next_states)
        # Then take only the non-terminal, valid ones for further expansion
        next_states = jax.tree.map(lambda x: x[nonzeros], next_states)
        construct_tree(next_states, depth + 1)
        
    
    construct_tree(init_state)

    efg = EFG(depth_isets=depth_isets,
              depth_history_legal=depth_history_legal,
              depth_history_actions=depth_history_actions,
              depth_history_utility=depth_history_utility,
              depth_history_next_history=depth_history_next_history,
              depth_history_player=depth_history_player,
              depth_history_is_chance=depth_history_is_chance,
              depth_history_chance_probabilities=depth_history_chance_probabilities,
              depth_iset_map=depth_iset_map,
              depth_iset_legal=depth_iset_legal,
              num_players=game.num_players,
              num_actions=game.num_actions,
              max_depth=len(depth_isets))
    return efg





def realization_plans(efg: EFG, depth_behaviorals: Iterable[np.ndarray]):
    """Propagates the infoset behavioral plans for all actions for each history.
    Also, the per player (infoset) reaches for each history.

    Args:
        efg (EFG): The extensive form game created by traverse_tree
        depth_behaviorals (Iterable[Iterable[np.ndarray]]): The current behaviorals.
        These are expected to provide the behavior strategy per infosets, with the same
        structure as the depth_iset_map of the EFG.
    """
    depth_reaches = []
    depth_realizations = []
    reaches = np.ones((efg.num_players + 1, 1))
    for d, d_behaviorals in enumerate(depth_behaviorals):
        cur_player = efg.depth_history_player[d]
        cur_iset = efg.depth_isets[d]
        is_chance = efg.depth_history_is_chance[d]
        chance_probs = efg.depth_history_chance_probabilities[d]
        d_behaviorals = [d_behaviorals[pl][iset] for pl, iset in zip(cur_player, cur_iset)]
        legal = efg.depth_history_legal[d]
        depth_reaches.append(reaches)
        player_realizations = [np.where((cur_player == pl)[..., None], d_behaviorals * reaches[pl][..., None] * legal, reaches[pl][..., None] * legal) for pl in range(efg.num_players)]
        chance_realizations = np.where(is_chance[..., None], reaches[-1][..., None] * chance_probs * legal, reaches[-1][..., None] * legal)
        realizations = np.stack([*player_realizations, chance_realizations], axis=0)
        depth_realizations.append(realizations)
        if d == efg.max_depth - 1:
            break
        next_nonterminal = efg.depth_history_next_history[d] >= 0
        reaches = np.stack([realizations[i, legal.astype(np.bool) * next_nonterminal].ravel() for i in range(efg.num_players + 1)], axis=0)
    return depth_reaches, depth_realizations

    
def uniform_strategy(efg: EFG):
    depth_behaviorals = []
    for d in range(efg.max_depth):
        iset_legals = efg.depth_iset_legal[d]
        uniform_behaviorals = [iset_legals[pl] / np.sum(iset_legals[pl], axis=-1, keepdims=True) for pl in range(efg.num_players)]
        depth_behaviorals.append(uniform_behaviorals)
    return depth_behaviorals


def evaluate(efg: EFG, depth_behaviorals: Iterable[np.ndarray]):
    """Compute the expected utility of each player in an extensive-form game."""

    reaches, realizations = realization_plans(efg, depth_behaviorals)
    expected_utils = np.zeros(efg.num_players)
    for d, d_realizations in enumerate(realizations):
        probs = np.prod(d_realizations, axis=0)
        weighted_utils = probs[..., None] * efg.depth_history_utility[d]
        #Flatten the utils such that player dimension is last, and everything else flattened
        weighted_utils = np.reshape(weighted_utils, (-1, efg.num_players))
        player_utils = np.sum(weighted_utils, axis=0)
        expected_utils += player_utils
    return expected_utils


def compute_best_response(efg: EFG, player: int, depth_behaviorals: Iterable[np.ndarray]):
    """Compute a given player against a fixed opponent strategy. Return that best
    response and its value

    Args:
        efg (EFG): The extensive form game
        player (int): Player to compute the best response value for
        depth_behaviorals (Iterable[np.ndarray]): The strategy profile represented as per depth
        iterable, where each depth has shape (num_players, infosets at depth). And it contains
        the behavioral strategies for each player at each infoset, structured in the same
        way as the EFG infoset map.

    Returns:
        br, br_value: The found best response. This has the form that it keeps the same strategies
        for all the other players as in the supplied behaviorals, and only replaces for the given
        player his strategies with his pure best responses. Also returns the value
        of that best response for the given player.
    """
    #First propagate reaches/realizations
    reaches, realizations = realization_plans(efg, depth_behaviorals)
    br_behaviorals = deepcopy(depth_behaviorals)
    num_last = np.size(efg.depth_history_next_history[-1])
    #Now we go bottom up to compute the best responses
    values = np.zeros(num_last)

    player_tup = (*range(efg.num_players), -1)
    for d in range(efg.max_depth - 1, -1, -1):
        num_histories = len(efg.depth_isets[d])
        new_values = np.zeros(num_histories)
        for pl in player_tup:
            #Just also including chance player
            player_mask = efg.depth_history_player[d] == pl if pl >= 0 else efg.depth_history_is_chance[d]
            if np.sum(player_mask) == 0:
                continue
            isets = efg.depth_isets[d][player_mask]
            history_behaviorals = depth_behaviorals[d][pl][isets] if pl >= 0 else efg.depth_history_chance_probabilities[d][player_mask]
            next_histories = efg.depth_history_next_history[d][player_mask].ravel()
            action_utilities = efg.depth_history_utility[d][player_mask]
            action_utilities = action_utilities[..., player].ravel()
            
            #make sure not the index by -1. The terminal histories will be filtered
            # away anyway
            #This indexing is just because
            # next histories contain the absolute indices
            # among all the histories.
            action_values = values[np.maximum(0, next_histories)]
            action_utilities = np.where(next_histories == -1, action_utilities, action_values)
            action_utilities = action_utilities.reshape(history_behaviorals.shape)
            #If the player is not the best responder, 
            # just take the current policy and propagate the value upwards
            # For chance this will always just weight it by chance probabilities
            if pl != player:
                depth_values = np.sum(action_utilities * history_behaviorals, axis=-1)
                new_values[player_mask] = depth_values
                continue
            #Now we compute the counterfactual reaches
            cf_reaches = np.where((np.arange(efg.num_players + 1) == pl)[..., None], 1, reaches[d])
            # Again, take only histories belonging to that player
            cf_reaches = np.prod(cf_reaches, axis=0)[player_mask]
            action_cf_utilities = action_utilities * cf_reaches[..., None]
            #Now convert from per history utilities to per iset utilites
            iset_utilities = np.bincount(efg.depth_history_actions[d][player_mask].ravel(), action_cf_utilities.ravel())
            iset_utilities = iset_utilities.reshape((-1, action_utilities.shape[-1]))
            #Make sure to prevent picking illegal actions for the best response
            iset_utilities = np.where(efg.depth_iset_legal[d][pl], iset_utilities, np.min(iset_utilities) - 1)
            #Save the pure best response
            iset_br = np.argmax(iset_utilities, axis=-1)
            #iset_first_br_val = np.max(iset_utilities, axis=-1)
            #If multiple actions have the same best response value,
            # choose the last one.
            #iset_brs = ((iset_utilities - iset_first_br_val[..., None]) ** 2) <= 1e-8
            #iset_brs = iset_brs * np.arange(iset_brs.shape[-1])[None,...]
            #iset_br = np.argmax(iset_brs ,axis=-1)
            iset_br_oh = np.eye(iset_utilities.shape[-1])[iset_br]
            br_behaviorals[d][player] = iset_br_oh
            history_br = iset_br[isets]
            # Finally, propagate the best response value upwards
            br_values=  np.take_along_axis(action_utilities, history_br[..., None], -1)
            new_values[player_mask] = np.squeeze(br_values, -1)
        values = new_values
    #The value at the root will form our best response value
    return br_behaviorals, values[0]




def compute_average_strategy(efg: EFG, behaviorals_first: Iterable[np.ndarray], behaviorals_second: Iterable[np.ndarray], 
                             t: int,  player:int = -1, linear = False):
    """Compute a weighted average of a pair of behavioural strategies.
    If a valid player is given (in range [0, num_players)), compute the averaging only for him"""
    reaches_first, realizations_first = realization_plans(efg, behaviorals_first)
    reaches_second, realizations_second = realization_plans(efg, behaviorals_second)
    averaged_behaviorals = []
    player_invalid = player < 0 or player >= efg.num_players
    d = 0
    for b_first, b_second, r_first, r_second in zip(behaviorals_first, behaviorals_second, reaches_first, reaches_second):
        cur_player = efg.depth_history_player[d]
        averaged_behaviorals.append([])
        #Get the reaches for the correct player
        #Change the reaches from per history to per iset
        for pl in range(efg.num_players):
            player_equal = player_invalid or pl == player
            mask = cur_player  == pl
            if not player_equal or np.sum(mask) == 0:
                #Just so the other players have something for the invalid iset as well to keep consistent shapes
                averaged_behaviorals[d].append(efg.depth_iset_legal[d][pl] / np.sum(efg.depth_iset_legal[d][pl], axis=-1, keepdims=True))
                continue
            isets = efg.depth_isets[d][mask].ravel()
            r_first, r_second = r_first[pl][mask], r_second[pl][mask]
            b_first, b_second = b_first[pl], b_second[pl]
            num_isets = b_first.shape[0]
            #Here we take advantage of the trick
            # that when multiple values are assigned to
            # the same index, the last one wins. 
            # We already have infoset reaches, so we
            # do NOT want to aggregate by bincount, 
            # otherwise we would multiply the reaches with the amount
            # of histories in them (due to perfect recall).
            # For the first, invalid infoset this will corectly keep zero reach.
            r_first_iset, r_second_iset = np.zeros(num_isets), np.zeros(num_isets)
            r_first_iset[isets], r_second_iset[isets] = r_first, r_second
            #for linear averaging, we weigh the component at timestep t
            # additionally by t, hence we linearly increase the weight
            # of the average components
            if linear:
                mult = 2 / (t + 2)
            else:
                mult = 1 / (t + 1)
            first_weighted_b =  (1 - mult) * r_first_iset[..., None] * b_first
            second_weighted_b =  mult * r_second_iset[..., None] * b_second
            pl_avg_behaviorals = first_weighted_b + second_weighted_b 
            #Now renormalize, except when the reaches are 0
            normalization = np.sum(pl_avg_behaviorals, axis=-1, keepdims=True)
            pl_avg_behaviorals = pl_avg_behaviorals / (normalization + (normalization == 0)) 
            averaged_behaviorals[d].append(pl_avg_behaviorals)
        d += 1
    return averaged_behaviorals

def nash_conv(efg: EFG, behaviorals: Iterable[np.ndarray]):
    """Compute NasConv (sum of incentives to deviate) for the given strategy profile.

    Args:
        efg (EFG): The extensive form game
        behaviorals (Iterable[np.ndarray]): The strategy profile represented as per depth
        iterable, where each depth has shape (num_players, infosets at depth). And it contains
        the behavioral strategies for each player at each infoset, structured in the same
        way as the EFG infoset map.
    """
    incentives_to_deviate = []
    for pl in range(efg.num_players):
        pl_br, pl_br_val = compute_best_response(efg, pl, behaviorals)
        incentives_to_deviate.append(pl_br_val)
    return sum(incentives_to_deviate)

def join_behaviorals(efg: EFG, per_player_behaviorals: Iterable[Iterable[np.ndarray]]) -> Iterable[np.ndarray]:
    """Join a sequence of behavioral strategies for each player into a single behavioral strategies.
    Such that strategy at infosets of player i is determined by the behavioral strategy at index i
    of the input iterable.

    Args:
        efg (EFG): The extensive form game
        per_player_behaviorals (Iterable[Iterable[np.ndarray]]): Behavioral strategies,
        sorted in ascending order based on which player they correspond to

    Returns:
        Iterable[np.ndarray]: The joint behavioral strategy
    """
    if len(per_player_behaviorals) == 1:
        return per_player_behaviorals[0]
    
    joint_behaviorals = []
    for d in range(efg.max_depth):
        player_d_behaviorals = [per_player_behaviorals[pl][d][pl] for pl in range(efg.num_players)]
        #Shape [num_players, Isets, actions]
        joint_behaviorals.append(player_d_behaviorals)
    return joint_behaviorals
    

def exploitability(efg: EFG, behaviorals: Iterable[np.ndarray]):
    """Compute exploitability of the given strategy profile. This is using
    the definition that exploitability = NashConv / num_players.

    Args:
        efg (EFG): The extensive form game
        behaviorals (Iterable[np.ndarray]): The strategy profile represented as per depth
        iterable, where each depth has shape (num_players, infosets at depth). And it contains
        the behavioral strategies for each player at each infoset, structured in the same
        way as the EFG infoset map.
    """
    return nash_conv(efg, behaviorals) / efg.num_players

def fictitous_play(efg: EFG, num_iters: int = 1000) -> Iterable[Iterable[np.ndarray]]:
    """Run the fictitous play algorithm for a given number of iterations.

    Args:
        efg (EFG): The extensive form game.
        num_iters (int, optional): Number of iterations. Defaults to 1000.

    Returns:
        Iterable[Iterable[np.ndarray]]: The sequence of average strategies for all players
        represented as joint behavioral strategy profiles.
    """
    strategies = []
    #Initialize average strategies uniformly
    avg_strategies = uniform_strategy(efg)
    #utilities = []
    #And initial strategies as best responses
    # to the uniform strategy of the oponents.
    last_strategies = join_behaviorals(efg, [compute_best_response(efg, pl, avg_strategies)[0] for pl in range(efg.num_players)])
    for it in range(num_iters):
        #utilities.append(evaluate(efg, avg_strategies))
        avg_strategies = compute_average_strategy(efg, avg_strategies, last_strategies, it + 1)
        last_strategies = join_behaviorals(efg, [compute_best_response(efg, pl, avg_strategies)[0] for pl in range(efg.num_players)])
        strategies.append(avg_strategies)
    return strategies




def compute_exploitability(efg: EFG, behaviorals_sequence: Iterable[Iterable[Iterable[np.ndarray]]], algorithm_name:str, save_dir: str = "plots/efg"):
    """Compute and plot the exploitability of a sequence of strategy profiles."""
    exploitabilities = []
    i = 0
    for b in behaviorals_sequence:
        expl = exploitability(efg, b)
        exploitabilities.append(expl)
        i += 1
    exploitabilities = np.asarray(exploitabilities)
    iters = np.arange(i)
    title_string = algorithm_name.replace("_", " ").title()
    plt.title(f"{title_string}")
    plt.xlabel("Iterations")
    plt.ylabel("Exploitability")
    plt.plot(iters, exploitabilities)
    plt_path = f"{save_dir}/{algorithm_name}.png"

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(plt_path)
    plt.close()

def env_walk(game: KuhnPoker):
    dummy_key = jax.random.key(0)
    state = game.init(dummy_key)
    num_actions = state.legal_action_mask.shape[0]
    # Pass the chance node first
    for i in range(2):
        action = jax.random.choice(dummy_key, num_actions, p=state.legal_action_mask / state.legal_action_mask.sum())
        state = game.step(state, action, dummy_key)
    first_action = jnp.array(0)
    second_action = jnp.array(1)
    past_first_action = game.step(state, first_action, dummy_key)
    past_second_action = game.step(state, second_action, dummy_key)

def main() -> None:

    # The implementation of the game is a part of a JAX library called `pgx`.
    # You can find more information about it here: https://www.sotets.uk/pgx/kuhn_poker/
    # We wrap the original implementation to add an explicit chance player and convert
    # everything from JAX arrays to Numpy arrays. There's also a JAX version which you
    # can import using `from kuhn_poker import KuhnPoker` if interested ;)
    env = KuhnPoker()
    #env_walk(env)

    efg = traverse_tree(env)
    #breakpoint()
    average_strategies = fictitous_play(efg, num_iters=1000)
    compute_exploitability(efg, average_strategies, "fictitous_play")

if __name__ == '__main__':
    main()
