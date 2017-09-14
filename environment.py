import numpy as np

class GridWorld:

    POSSIBLE_ACTIONS = ['U', 'D', 'L', 'R']
    WIND = [0.8, 0.1, 0.1]

    def __init__(self, size, start, end, rewards, actions):
        """Initializes GridWorld environment

        Parameters
        ----------
        size: A tuple (h,w) of height and width of the grid
        start: A tuple (i,j) of starting state in the grid
        rewards: A dict of mapping from state to rewards {(i,j): int}
        actions: A dict of mapping from state to list of actions {(i,j): list}

        """

        self.height, self.width = size
        self.set_state(start)
        self.end = end

        self.rewards = rewards
        self.actions = actions

        self.num_states = np.prod(size)
        self.num_actions = len(GridWorld.POSSIBLE_ACTIONS)
        
    
    def set_state(self,state):
        self.i, self.j = state

    def current_state(self):
        return (self.i, self.j)
    
    def move_state(self, action, state):
        # check if legal action
        if action in self.actions[self.current_state()]:
            i, j = state
            if action == 'U': self.set_state((i-1,j))
            elif action == 'D': self.set_state((i+1,j))
            elif action == 'R': self.set_state((i,j+1))
            elif action == 'L': self.set_state((i,j-1))

    def transition(self, action, state = None):
        # returns [prob, new_state, reward, is_done]
        assert(action in GridWorld.POSSIBLE_ACTIONS)
        prob = 0.8
        
        # based on probabilities choose stochastic actions
        if action == 'U':
            sample = np.random.choice(['U','R','L'], 1, p=GridWorld.WIND)
            if(sample[0] != 'U'): prob = 0.1
        elif action == 'D':
            sample = np.random.choice(['D','R','L'], 1, p=GridWorld.WIND)
            if(sample[0] != 'D'): prob = 0.1
        elif action == 'R':
            sample = np.random.choice(['R','U','D'], 1, p=GridWorld.WIND)
            if(sample[0] != 'R'): prob = 0.1
        elif action == 'L':
            sample = np.random.choice(['L','U','D'], 1 ,p=GridWorld.WIND)
            if(sample[0] != 'L'): prob = 0.1
        
        # if state is not given in args use current state as the source
        if state == None: source = self.current_state()
        else: source = state
        
        self.move_state(action=sample[0], state=source)

        # are we at the goal state
        is_done = self.current_state() == self.end
        
        return [(prob, self.current_state(), self.rewards.get((self.current_state()),0), is_done)]

def grid():

    rewards = {
        (0,0) : -1,(0,1) : -1,(0,2) : -1,(0,3) : -1,
        (1,0) : -1,(1,1) : -1,(1,2) : -1,(1,3) : -1, # start state is 1x0
        (2,0) : -1,(2,1) : -70,(2,2) : -1,(2,3) : -1, # bad state is 2x1
        (3,0) : -1,(3,1) : -1,(3,2) : -1,(3,3) : 100 # goal state is 3x3
    }

    actions = {
        (0,0) : ['R','D'], (0,1) : ['R','L','D'],
        (0,2) : ['R','L','D'], (0,3) : ['L','D'],
        (1,0) : ['R','U','D'], (1,1) : ['R','L','U','D'],
        (1,2) : ['R','L','U','D'], (1,3) : ['L','U','D'],
        (2,0) : ['R','U','D'], (2,1) : ['R','L','U','D'],
        (2,2) : ['R','L','U','D'],(2,3) : ['L','U','D'],
        (3,0) : ['R','U'], (3,1) : ['R','L','U'],
        (3,2) :['R','L','U']
    }

    return GridWorld( size=(4,4),start=(1,0), end=(3,3), \
                    rewards = rewards, actions = actions )