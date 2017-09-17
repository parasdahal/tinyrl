import numpy as np


all_actions = ['U','D','L','R']

def epsilon_greedy(action,epsilon):
    p = np.random.random()
    if p < (1 - epsilon):
        return action
    else: 
        return np.random.choice(all_actions,1)[0]

def q_learning(env, num_episodes, epsilon, alpha, start, stop, discount_factor=0.99):
    
    all_states = set(env.rewards.keys())
    
    Q = {}
    for state in all_states:
        Q[state] = {}
        for action in all_actions:
            Q[state][action] = 0
    
    average_q = []

    for episode in range(num_episodes):
        print("Episode:",episode)
        state = start
        average_q.append(0)
        while state != stop:
            
            action = max(Q[state], key = Q[state].get)
            action = epsilon_greedy(action, epsilon)
            new_state, reward = env.transition(action,state,choose=True)
            best_next_action = max(Q[new_state],key=Q[new_state].get)

            average_q[episode] += reward
            
            td_target = reward + discount_factor * Q[new_state][best_next_action]
            Q[state][action] += alpha * (td_target - Q[state][action])

            state = new_state
        
        for state in all_states:
            average_q[episode] += Q[state][max(Q[state], key = Q[state].get)]
        average_q[episode] /= 15
    

    policy, V = {}, {}
    for state in all_states:
        best_a = max(Q[state], key=Q[state].get)
        best_q = Q[state][best_a]
        if state == stop: 
            policy[state] = 'G'
        else: policy[state] = best_a
        V[state] = best_q
    
    return policy, V, average_q
            



