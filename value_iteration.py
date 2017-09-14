import numpy as np
from environment import grid

env = grid()

POSSIBLE_ACTIONS = ['U', 'D', 'L', 'R']

def intialize(env):
    policy, V = {}, {}
    
    for state in env.actions.keys():
        policy[state] = np.random.choice(POSSIBLE_ACTIONS)
    
    all_states = set(env.rewards.keys())
    for state in all_states:
        if state in env.actions:
            V[state] = np.random.random()
        else:
            V[state] = 0
    return policy, V

def calculate_v(V, state, discount_factor = 0.99):
    expected_values = {'U':0,'D':0,'L':0,'R':0}
    for action in POSSIBLE_ACTIONS:
        for prob, next_state, reward, done in env.transition(action,state):
            expected_values[action] += prob * (reward + discount_factor* V[next_state])
            
    best_v = max(expected_values,key=expected_values.get)
    best_a = expected_values[best_v]
    return best_v,best_a

def value_iteration(env, policy, V, discount_factor = 0.99, threshold = 0.001):
    all_states = set(env.rewards.keys())
    # run until convergence
    while True:
        delta = 0
        for state in all_states:
            # calculate value of all the actions in this state
            best_v, _ = calculate_v(V,state)
            delta = max(delta, np.abs(best_v,V[state]))
            V[state] = best_v
        if delta < threshold:
            break
    
    for state in all_states:
        _, best_a = calculate_v(V,state)
        policy[state] = best_a
    
    return policy, V

            

            

    