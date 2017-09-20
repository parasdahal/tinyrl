import numpy as np
from environment import play_game


def value_iteration(env, start, stop, discount_factor=0.99, threshold=0.001):
    """Performs Value Iteartion algorithm on given environment

    Parameters
    ----------
    env : GridWorld
        The GridWorld environment object
    start : (row, col)
        The start state of the grid
    stop : (row, col)
        The end state of the grid
    discount_factor : float
        Factor that represents care for future rewards
    threshold: float
        Value that represents cutoff for iterations

    Returns
    -------
    policy: dict {(row, col): char}
        The optimal policy dict with optimal action for each state
    V: dict {(row, col): float}
        The optimal value dict with optimal values for each state
    stats: list of {policy:list,score:list}
        The policy and V values for each iteration, used for visualization
    """

    def calculate_v(V, state, actions):
        """ V[s] = max[a]{ sum[s',r] { p(s',r|s,a)[r + gamma*V[s']] } }
        """
        new_V = {'U': 0, 'D': 0, 'L': 0, 'R': 0}

        for action in actions:
            # get all possible transitions list
            possible_transitions = env.transition(action, state)
            for prob, next_state, reward in possible_transitions:
                new_V[action] += prob * \
                    (reward + discount_factor * V[next_state])

        # key with max value
        best_a = max(new_V, key=new_V.get)
        # max value in the dict
        best_V = new_V[best_a]

        return best_V, best_a

    all_states = set(env.rewards.keys())
    # Initialize V to 0
    V = {}
    for state in all_states:
        V[state] = 0

    episode = 0
    stats = []

    # run until convergence
    while True:

        stats.append({})
        delta = 0

        for state in all_states:
            best_V, _ = calculate_v(V, state, env.actions[state])
            delta = max(delta, np.abs(best_V - V[state]))
            V[state] = best_V

        # collect stats for visuaization
        policy_list, v_list = [], []
        # iterate on sorted states as policy and v are a list with
        # index representing the states
        for state in sorted(all_states):
            best_v, best_a = calculate_v(V, state, env.actions[state])
            policy_list.append(best_a)
            v_list.append(best_v)

        # stats for this iteration
        stats[episode] = {'policy': policy_list, 'score': v_list}

        episode += 1

        # check if converged
        if delta < threshold:
            break

    # get optimal policy
    policy = {}
    for state in all_states:
        _, best_a = calculate_v(V, state, env.actions[state])
        policy[state] = best_a

    return policy, V, stats
