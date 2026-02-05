#!/usr/bin/env python3
from templates.week07 import *


from copy import deepcopy

f32 = np.float32

def regret_matching(regrets: np.ndarray):
    """Computes current strategy given cumulative regrets.

    Args:
        regrets (np.ndarray): The array of regrets. Assumed to be just for single player and single depth (so shape [S(D), A])
    Returns:
        The current strategy induced by the cumulative regrets
    """
    num_actions = regrets.shape[-1]
    pos_regrets = regrets * (regrets > 0)
    normalization = np.sum(pos_regrets, axis=-1, keepdims=True)
    zero_normalization = np.asarray(normalization == 0, dtype=f32)
    normalized_regrets = pos_regrets / (normalization + zero_normalization)
    uniform_strat = np.ones_like(normalized_regrets) / num_actions
    #To handle correctly returning a uniform strategy under all zero cumulative regrets
    strategy = (1 - zero_normalization) * normalized_regrets + zero_normalization * uniform_strat
    return strategy

def cfr_depth_update(efg: EFG,depth_reaches: np.ndarray, next_depth_values: np.ndarray,
                      depth_strategies: list[np.ndarray], d: int,  player: int = -1):
    """Performs the CFR update logic for a single depth

    Args:
        efg (EFG): The extensive form game
        depth_reaches (np.ndarray): Reaches for the given depth of shape (Pl + 1, H(D))
        next_depth_values (np.ndarray): Values (not counterfactually weighted) for 
         the next_histories of shape (H(D + 1),Pl)
        depth_strategies (np.ndarray): Per player current strategies for the given depth.
        Of the shame 
        d (int) : The given depth
        player (int): The given player for who to update. If < 0, updates for all (default behavior)

    Returns:
     Values for the current depth of shape (Pl, H(D)), Regrets for the current depth, of the same
     structure as depth_iset_legals for the current depth.
    """
    
    simultaneous_update = player < 0
    H_d = efg.depth_isets[d].shape[0]

    current_actions = efg.depth_history_actions[d]
    next_histories = efg.depth_history_next_history[d]
    all_action_utilities = efg.depth_history_utility[d]
    new_values = np.zeros((H_d, efg.num_players))

    #-1 is the chance player
    player_tup = (*range(efg.num_players), -1)

    regrets = [np.zeros(efg.depth_iset_legal[d][pl].shape, dtype=f32) for pl in range(efg.num_players)]
    for pl in player_tup:
        #To handle chance player
        player_mask = efg.depth_history_player[d] == pl if pl >= 0 else efg.depth_history_is_chance[d]
        if np.sum(player_mask) == 0:
            continue

        pl_actions = current_actions[player_mask]
        pl_next_histories = next_histories[player_mask][..., None]
        action_utilities = all_action_utilities[player_mask]

        action_values = next_depth_values[np.maximum(0, pl_next_histories.ravel())]
        action_values = action_values.reshape((np.sum(player_mask), efg.num_actions, efg.num_players))
        action_utilities = np.where(pl_next_histories == -1, action_utilities, action_values)
        #Propagate the value upwards
        player_iset = efg.depth_isets[d][player_mask]
        per_history_strategies = depth_strategies[pl][player_iset] if pl >= 0 else efg.depth_history_chance_probabilities[d][player_mask]
        new_values[player_mask] = np.sum(action_utilities * per_history_strategies[..., None], axis=-2)
        #For alternating updates, do not update
        # the regrets for the other players
        # also, if pl < 0 it is the chance player.
        # do not update anything for him.
        if (pl != player and not simultaneous_update):
            continue
        #Now we compute the counterfactual reaches
        cf_reaches = np.where((np.arange(efg.num_players + 1) == pl)[..., None], 1, depth_reaches)
        # Again, take only histories belonging to that player
        cf_reaches = np.prod(cf_reaches, axis=0)[player_mask]
        action_cf_utilities = action_utilities[..., pl] * cf_reaches[..., None]
        iset_action_utilities = np.bincount(pl_actions.ravel(), action_cf_utilities.ravel())
        iset_action_utilities = iset_action_utilities.reshape((-1, efg.num_actions))
        #compute the regrets
        iset_utilities = np.sum(iset_action_utilities * depth_strategies[pl], axis=-1, keepdims = True)
        #Do not allow any regret on illegal actions
        cf_regrets = (iset_action_utilities - iset_utilities) * efg.depth_iset_legal[d][pl]
        regrets[pl] = cf_regrets
    return regrets, new_values


        
        


def cfr(efg: EFG, num_iterations:int) -> list[list[list[np.ndarray]]]:
    """Run the CFR algorithm for a given number of iterations.
    Return the list of average behaviorals at each iteration."""

    cumulative_regrets = [[np.zeros(pl.shape, dtype=f32) for pl in d] for d in efg.depth_iset_legal]
    
    cumulative_strategies = [[np.zeros(pl.shape, dtype=f32) for pl in d] for d in efg.depth_iset_legal]

    current_strategies = normalize_strategy(efg, cumulative_strategies)
    per_iter_averages = [current_strategies]

    for it in range(num_iterations):
        #Propagate the reaches and realization plans
        # per history of the current strategt
        reaches, realizations = realization_plans(efg, current_strategies)
        #Now iterate from bottom up, to compute the counterfactual regrets and new strategy
        # We propagate the value upwards (NOT counterfactually weighted), similarly as during best
        # response computation
        num_last = np.size(efg.depth_history_next_history[-1])
        values = np.zeros((num_last, efg.num_players))
        for d in range(efg.max_depth - 1, -1, -1):
            regrets, values = cfr_depth_update(efg, reaches[d], values, current_strategies[d], d, player=-1)
            for pl in range(efg.num_players):
                cumulative_regrets[d][pl] += regrets[pl]
                current_strategies[d][pl] = regret_matching(cumulative_regrets[d][pl])
        cumulative_strategies = update_cumulative_strategy(efg, cumulative_strategies, current_strategies, it + 1)
        per_iter_averages.append(normalize_strategy(efg, cumulative_strategies))
    return per_iter_averages    



def cfr_plus(efg: EFG, num_iterations: int) -> list[list[list[np.ndarray]]]:
    """Run the CFR+ algorithm for a given number of iterations.
    Return the list of average behaviorals at each iteration."""

    cumulative_regrets = [[np.zeros(pl.shape, dtype=f32) for pl in d] for d in efg.depth_iset_legal]
    
    cumulative_strategies = [[np.zeros(pl.shape, dtype=f32) for pl in d] for d in efg.depth_iset_legal]
    

    current_strategies = normalize_strategy(efg, cumulative_strategies)
    per_iter_averages = [current_strategies]

    for it in range(num_iterations):
        #Alternating updates, always update
        # for a single player and then propagate 
        # the realization plans again
        for pl in range(efg.num_players):
            #Propagate the reaches and realization plans
            # per history of the current strategt
            reaches, realizations = realization_plans(efg, current_strategies)
            #Now iterate from bottom up, to compute the counterfactual regrets and new strategy
            # We propagate the value upwards (NOT counterfactually weighted), similarly as during best
            # response computation
            num_last = np.size(efg.depth_history_next_history[-1])
            values = np.zeros((num_last, efg.num_players))
            for d in range(efg.max_depth - 1, -1, -1):
                regrets, values = cfr_depth_update(efg, reaches[d], values, current_strategies[d], d, player=pl)
                # Regret matching+, clamp the cumulative sum to zero
                cumulative_regrets[d][pl] = np.maximum(0, cumulative_regrets[d][pl] + regrets[pl])
                current_strategies[d][pl] = regret_matching(cumulative_regrets[d][pl])
        #Make sure to use the linear averaging
        cumulative_strategies = update_cumulative_strategy(efg, cumulative_strategies, current_strategies, it + 1, multcoeff=1.0)
        per_iter_averages.append(normalize_strategy(efg, cumulative_strategies))
    return per_iter_averages    

def test_kuhn():
    kuhn = KuhnPoker()
    efg = traverse_tree(kuhn)
    cfr_averages = cfr(efg, num_iterations=500)
    compute_exploitability(efg, cfr_averages, "CFR_kuhn")
    cfr_plus_averages = cfr_plus(efg, num_iterations=500)
    compute_exploitability(efg, cfr_plus_averages, "CFR+_kuhn")

def test_leduc():
    leduc = LeducPoker()
    efg = traverse_tree(leduc)
    cfr_averages = cfr(efg, num_iterations=500)
    compute_exploitability(efg, cfr_averages, "CFR_leduc")
    cfr_plus_averages = cfr_plus(efg, num_iterations=500)
    compute_exploitability(efg, cfr_plus_averages, "CFR+_leduc")

def main() -> None:
    test_kuhn()
    test_leduc()
    


if __name__ == '__main__':
    main()
