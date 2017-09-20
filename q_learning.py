import numpy as np
from environment import play_game

all_actions = ['U', 'D', 'L', 'R']


def _epsilon_greedy(action, epsilon):
    """Returns action according to epsilon greedy exploration scheme
    """
    p = np.random.random()
    if p < (1 - epsilon):
        return action
    else:
        return np.random.choice(all_actions, 1)[0]


def q_learning(env, num_episodes, epsilon, alpha,
               start, stop, discount_factor=0.99):
    """Performs q learning algorithm on given environment

    Parameters
    ----------
    env : GridWorld
        The GridWorld environment object
    num_episodes: int
        The number of episodes to run
    epsilon: float
        Epsilon value for exploration
    alpha: float
        Learning rate
    start : (row, col)
        The start state of the grid
    stop : (row, col)
        The end state of the grid
    discount_factor : float
        Factor that represents care for future rewards

    Returns
    -------
    policy: dict {(row, col): char}
        The optimal policy dict with optimal action for each state
    V: dict {(row, col): float}
        The optimal value dict with optimal values for each state
    stats: list of {policy:list,score:list}
        The policy and V values for each iteration, used for visualization
    """

    all_states = set(env.rewards.keys())

    # initialize Q[s][a]
    Q = {}
    for state in all_states:
        Q[state] = {}
        for action in all_actions:
            Q[state][action] = 0

    stats = []
    # repeat for number of episodes
    for episode in range(num_episodes):
        state = start
        stats.append({})
        # until end state is reached
        while state != stop:
            # take best action for current state and use epsilon greedy
            action = max(Q[state], key=Q[state].get)
            action = _epsilon_greedy(action, epsilon)

            # using above action, find next state and best action for that
            # state
            new_state, reward = env.transition(action, state, choose=True)
            best_next_action = max(Q[new_state], key=Q[new_state].get)

            # calculate td target and update Q[s][a]
            td_target = reward + discount_factor * \
                Q[new_state][best_next_action]
            Q[state][action] += alpha * (td_target - Q[state][action])

            state = new_state

        # collect stats
        policy_list, q_list = [], []
        # iterate on sorted states as policy and v are a list with
        # index representing the states
        for state in sorted(all_states):
            best_a = max(Q[state], key=Q[state].get)
            best_q = Q[state][best_a]
            if state != stop:
                policy_list.append(best_a)
            else:
                policy_list.append('G')
            q_list.append(best_q)

        # add steps according to current policy to visualize agent's actions
        stats[episode] = {
            'policy': policy_list,
            'score': q_list,
            'steps': play_game(
                env,
                start,
                stop,
                policy_list)}

    # optimal policy and V
    policy, V, = {}, {}
    for state in all_states:
        best_a = max(Q[state], key=Q[state].get)
        best_q = Q[state][best_a]
        policy[state] = best_a if state != stop else 'G'
        V[state] = best_q

    return policy, V, stats
