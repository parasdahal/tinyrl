import numpy as np
from environment import play_game

all_actions = ['U', 'D', 'L', 'R']


def _epsilon_greedy(action, epsilon):
    p = np.random.random()
    if p < (1 - epsilon):
        return action
    else:
        return np.random.choice(all_actions, 1)[0]


def q_learning(env, num_episodes, epsilon, alpha,
               start, stop, discount_factor=0.99):

    all_states = set(env.rewards.keys())

    Q = {}
    for state in all_states:
        Q[state] = {}
        for action in all_actions:
            Q[state][action] = 0

    stats = []

    for episode in range(num_episodes):
        state = start
        stats.append({})
        while state != stop:

            action = max(Q[state], key=Q[state].get)
            action = _epsilon_greedy(action, epsilon)
            new_state, reward = env.transition(action, state, choose=True)
            best_next_action = max(Q[new_state], key=Q[new_state].get)
            
            
            td_target = reward + discount_factor * \
                Q[new_state][best_next_action]
            Q[state][action] += alpha * (td_target - Q[state][action])

            state = new_state
        
        # collect stats
        policy_list, q_list = [],[]
        for state in sorted(all_states):
            best_a = max(Q[state], key=Q[state].get)
            best_q = Q[state][best_a]
            if state != stop: policy_list.append(best_a)
            else: policy_list.append('G')
            q_list.append(best_q)
        
        stats[episode] = {'policy':policy_list,'score':q_list,'steps':play_game(env,start,stop,policy_list)}

    policy, V, = {}, {}

    for state in all_states:
        best_a = max(Q[state], key=Q[state].get)
        best_q = Q[state][best_a]
        policy[state] = best_a if state != stop else 'G'
        V[state] = best_q

    return policy, V, stats
