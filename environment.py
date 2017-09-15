import numpy as np

class GridWorld:

    POSSIBLE_ACTIONS = ['U', 'D', 'L', 'R']

    def __init__(self, size, rewards, actions):
        """Initializes GridWorld environment

        Parameters
        ----------
        size: A tuple (h,w) of height and width of the grid
        start: A tuple (i,j) of starting state in the grid
        rewards: A dict of mapping from state to rewards {(i,j): int}
        actions: A dict of mapping from state to list of actions {(i,j): list}

        """

        self.height, self.width = size

        self.rewards = rewards
        self.actions = actions

        self.num_states = np.prod(size)
        self.num_actions = len(GridWorld.POSSIBLE_ACTIONS)

    
    def _limit_coordinates(self,state):
        i, j = state
        if i < 0: i = 0
        elif i > self.height-1: i = self.height - 1
        if j < 0: j = 0
        elif j > self.width-1: j = self.width - 1
        return (i,j)


    def _new_state_reward(self, action, state):
        # check if legal action
        
        i, j = state
        if action == 'U': i,j = i-1,j
        elif action == 'D': i,j = i+1,j
        elif action == 'R': i,j = i,j+1
        elif action == 'L': i,j = i,j-1
        new_state = self._limit_coordinates((i,j))        
        return new_state, self.rewards.get(new_state)

    def transition(self, action, state):

        def stochastic(possible_actions,prob):
            result = []
            for i,a in enumerate(possible_actions):
                coord, reward = self._new_state_reward(a, state)
                result.append((prob[i], coord, reward))
            return result

        if action == 'U': return stochastic(['U','R','L'],[0.8,0.1,0.1])
        elif action == 'D': return stochastic(['D','R','L'],[0.8,0.1,0.1])
        elif action == 'R': return stochastic(['R','U','D'],[0.8,0.1,0.1])
        elif action == 'L': return stochastic(['L','U','D'],[0.8,0.1,0.1])
        

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
        (3,2) :['R','L','U'], (3,3) : []
    }

    return GridWorld( size=(4,4), rewards = rewards, actions = actions )