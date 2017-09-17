import numpy as np


def value_iteration(env, discount_factor=0.99, threshold=0.001):

    def calculate_v(V, state, actions):
        # V[s] = max[a]{ sum[s',r] { p(s',r|s,a)[r + gamma*V[s']] } }
        new_V = {'U': 0, 'D': 0, 'L': 0, 'R': 0}

        for action in actions:
            possible_transitions = env.transition(action, state)
            for prob, next_state, reward in possible_transitions:
                new_V[action] += prob * \
                    (reward + discount_factor * V[next_state])

        best_a = max(new_V, key=new_V.get)
        best_V = new_V[best_a]

        return best_V, best_a

    all_states = set(env.rewards.keys())
    V = {}
    for state in all_states:
        V[state] = 0
    
    episode = 0
    stats = []

    while True:
        stats.append({})
        delta = 0
        for state in all_states:
            best_V, _ = calculate_v(V, state, env.actions[state])
            delta = max(delta, np.abs(best_V - V[state]))
            V[state] = best_V
        
        # collect stats
        policy_list, v_list = [],[]
        for state in sorted(all_states):
            best_v, best_a = calculate_v(V, state, env.actions[state])
            policy_list.append(best_a)
            v_list.append(best_v)
        
        stats[episode] = {'policy':policy_list,'score':v_list}
        
        episode += 1
        
        if delta < threshold:
            break

    policy = {}
    for state in all_states:
        _, best_a = calculate_v(V, state, env.actions[state])
        policy[state] = best_a

    return policy, V, stats