#!/usr/bin/env python3
import numpy as np

from templates.week07 import *
from templates.week09 import cfr_depth_update, regret_matching


f32 = np.float32
f64 = np.float64


def discounted_cfr(efg: EFG, num_iterations: int, alpha:float=1.5, beta:float=0.0, gamma=2.0) -> list[list[list[np.ndarray]]]:
    """Run the Discounted CFR algorithm for a given number of iterations.

    Args:
        efg (EFG): The game
        num_iterations (int): The number of iterations to run
        alpha (float, optional): The weighing factor of positive regrets. Defaults to 1.5 (recent
        positive regrets are upweighted moderately.).
        beta (float, optional): The weighing factor of negative regrets. Defaults to 0.0 (negative
        regrets are weighted uniformly).
        gamma (float, optional): The weighing factor of average strategy. Defaults to 2.0
        (Recent strategies are upweighted quadratically).

    Returns:
        list[list[list[np.ndarray]]]: _description_
    """
    cumulative_regrets = [[np.zeros(pl.shape, dtype=f32) for pl in d] for d in efg.depth_iset_legal]
    
    cumulative_strategies = [[np.zeros(pl.shape, dtype=f32) for pl in d] for d in efg.depth_iset_legal]

    current_strategies = normalize_strategy(efg, cumulative_strategies)
    per_iter_averages = [current_strategies]

    for it in range(num_iterations):
        positive_regret_mult = (it + 1) ** alpha
        negative_regret_mult = (it + 1) ** beta
        positive_regret_mult = positive_regret_mult / (positive_regret_mult + 1)
        negative_regret_mult = negative_regret_mult / (negative_regret_mult + 1)
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
                negative_regrets = cumulative_regrets[d][pl] < 0
                positive_regrets = 1 - negative_regrets
                negative_force = cumulative_regrets[d][pl] * negative_regret_mult + regrets[pl]
                positive_force = cumulative_regrets[d][pl] * positive_regret_mult + regrets[pl]
                cumulative_regrets[d][pl] = positive_force * positive_regrets + negative_force * negative_regrets
                current_strategies[d][pl] = regret_matching(cumulative_regrets[d][pl])
        #Make sure to use the linear averaging
        cumulative_strategies = update_cumulative_strategy(efg, cumulative_strategies, current_strategies, it + 1, multcoeff=gamma)
        per_iter_averages.append(normalize_strategy(efg, cumulative_strategies))
    return per_iter_averages 

    


def monte_carlo_cfr(efg: EFG, num_iterations: int, seed:int = 42, eps: float=0.1):
    """Run the Monte Carlo CFR algorithm with outcome sampling  for a given number of iterations.

    Args:
        efg (EFG): The game
        num_iterations (int): Amount of iterations
        seed(int): The RNG seed
        eps (float, optional): The strength of the uniform mixture in the sampling policy. Defaults to 0.01.
    """

    cumulative_regrets = [[np.zeros(pl.shape, dtype=f32) for pl in d] for d in efg.depth_iset_legal]
    
    
    cumulative_strategies = [[np.zeros(pl.shape, dtype=f32) for pl in d] for d in efg.depth_iset_legal]
    gen = np.random.default_rng(seed=seed)
    current_strategies = normalize_strategy(efg, cumulative_strategies)

    per_iter_averages = [current_strategies]
    def sample_outcome(depth:int, history:int, cur_reaches:np.ndarray, cur_sample_prob: float):
        """Helper method to sample an action and return the 
        sampled outcome, the outcome probability, infoset index, the updated reaches,
         and the updated_probability under the sampling policy. Returns -1 in place of the infoset index"""
        player = efg.depth_history_player[depth][history]
        legals = efg.depth_history_legal[depth][history]
        new_reaches = cur_reaches.copy()
        #Handle chance nodes
        if efg.depth_history_is_chance[depth][history]:
            #Just to be sure, multiply by legal and renormalize
            chance_probs = efg.depth_history_chance_probabilities[depth][history]
            chance_probs = chance_probs * legals
            chance_probs = chance_probs / np.sum(chance_probs, axis=-1, keepdims=True)
            outcome = gen.choice(legals.shape[-1], p=chance_probs)
            outcome_prob = chance_probs[outcome]
            new_reaches[-1] *= outcome_prob
            return outcome, outcome_prob, -1, new_reaches, cur_sample_prob * outcome_prob
        cur_iset = efg.depth_isets[depth][history]
        policy = current_strategies[depth][player][cur_iset]
        uniform = legals / np.sum(legals, axis=-1, keepdims=True, dtype=f32)
        sampling_policy = (1 - eps) * policy + eps * uniform
        #Renormalize so that numpy does not complain about
        #numerical inaccuracies
        sampling_policy = sampling_policy / np.sum(sampling_policy, axis=-1, keepdims=True)
        try:
            outcome = gen.choice(legals.shape[-1], p=sampling_policy)
        except ValueError:
            print(f"Legals: {efg.depth_history_legal[depth]}")
            breakpoint()
        outcome_prob = policy[outcome]
        new_reaches[player] *= outcome_prob
        return outcome, outcome_prob, cur_iset, new_reaches, cur_sample_prob * sampling_policy[outcome]
    def sample_history(player: int):
        """Sample one terminal history and perform regret
        updates for infosets along it for a particular player"""
        #Collect the data along the sampled history
        reaches = np.ones(efg.num_players + 1, dtype=f32)
        history_idx = 0
        history_reaches = [reaches]
        history_indices = [history_idx]
        action_indices = []
        iset_indices = []
        outcome_probs = []
        history_return = 0
        sample_prob = 1
        for d in range(efg.max_depth):
            action, prob, iset, reaches, sample_prob = sample_outcome(d, history_idx, reaches, sample_prob)
            outcome_probs.append(prob)
            action_indices.append(action)
            iset_indices.append(iset)
            history_return += efg.depth_history_utility[d][history_idx][action]
            history_idx = efg.depth_history_next_history[d][history_idx][action]
            #If we are at a terminal level,
            # do not store the next history and the next
            # history reach
            if d == efg.max_depth - 1 or history_idx < 0:
                break
            history_indices.append(history_idx)
            history_reaches.append(reaches)
        #Second step, perform the regret updates
        history_len = len(history_reaches)
        for d in range(history_len - 1, -1, -1):
            iset, action, history_idx, reaches  = iset_indices[d], action_indices[d], history_indices[d], history_reaches[d]
            history_player = efg.depth_history_player[d][history_idx]
            #0 or lower is an invalid infoset. So, this is
            # a chance level. Skip the update.
            # Same if the player is not acting here
            if iset <= 0 or history_player != player:
                continue
            #Use the update rule as detailed
            # in the supplemental material of the original paper
            # https://papers.nips.cc/paper_files/paper/2009/hash/00411460f7c92d2124a67ea0f4cb5f85-Abstract.html
            cf_history_reach = np.prod(np.where(np.arange(efg.num_players + 1) == player, 1, reaches))
            W = (history_return[player] * cf_history_reach) / sample_prob
            #Reaches of the terminal history
            # history/action, history
            terminal_from_history_action = 1 if d == history_len - 1 else np.prod(outcome_probs[d + 1:])
            terminal_from_history = terminal_from_history_action * outcome_probs[d]
            #Update all actions that are available at the infoset
            # but zero out the action values for 
            # the ones that were not played
            action_mask = np.arange(efg.depth_iset_legal[d][player].shape[-1]) == action
            regrets = W * ((terminal_from_history_action * action_mask) - terminal_from_history)
            #Do not forget to mask out illegal actions
            regrets = regrets * efg.depth_iset_legal[d][player][iset]
            cumulative_regrets[d][player][iset] += regrets
            current_strategies[d][player][iset] = regret_matching(cumulative_regrets[d][player][iset])
            #print(f"Depth {d}")
            #print(f"Cf history reach {cf_history_reach}")
            #print(f"Regrets {regrets}")
            #breakpoint()
    
    for it in range(num_iterations):
        for pl in range(efg.num_players):
            sample_history(pl)
            #print(f"Current strategies {current_strategies}")
        cumulative_strategies = update_cumulative_strategy(efg, cumulative_strategies, current_strategies, it + 1, multcoeff=1.0)
        per_iter_averages.append(normalize_strategy(efg, cumulative_strategies))
    return per_iter_averages
        
            
            

def kuhn_test():
    kuhn = KuhnPoker()
    efg = traverse_tree(kuhn)
    dcfr_averages = discounted_cfr(efg, num_iterations=500, alpha=1.5, beta=0, gamma=2)
    compute_exploitability(efg, dcfr_averages, "DCFR_kuhn")
    mccfr_averages = monte_carlo_cfr(efg, num_iterations=500, seed=42, eps=0.01)
    compute_exploitability(efg, mccfr_averages, "MCCFR_kuhn")

def leduc_test():
    kuhn = LeducPoker()
    efg = traverse_tree(kuhn)
    dcfr_averages = discounted_cfr(efg, num_iterations=500, alpha=1.5, beta=0, gamma=2)
    compute_exploitability(efg, dcfr_averages, "DCFR_leduc")
    mccfr_averages = monte_carlo_cfr(efg, num_iterations=500, seed=42, eps=0.01)
    compute_exploitability(efg, mccfr_averages, "MCCFR_leduc")




def main() -> None:
    kuhn_test()
    leduc_test()


if __name__ == '__main__':
    main()
