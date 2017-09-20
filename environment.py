import numpy as np


class GridWorld:
    """
    Gird environment with following stochastic property:

    | Agent Action | Possible Actions  |  Probability  |
    | :----------: | :---------------: | :-----------: |
    |      UP      |  UP, RIGHT, LEFT  | 0.8, 0.1, 0.1 |
    |     DOWN     | DOWN, RIGHT, LEFT | 0.8, 0.1, 0.1 |
    |     LEFT     |  LEFT, UP, DOWN   | 0.8, 0.1, 0.1 |
    |    RIGHT     |  RIGHT, UP, DOWN  | 0.8, 0.1, 0.1 |
    """

    POSSIBLE_ACTIONS = ['U', 'D', 'L', 'R']

    def __init__(self, size, rewards, actions):
        """Initialize the GridWorl object

        Parameters
        ----------
        size : tuple (row,col)

        rewards: dict {(row,col):int}
            A dictionary with reward values for each state in the grid

        actions: dict ({row,col}:list)
            A dictionary that associates all possible actions for each state

        """

        self.height, self.width = size

        self.rewards = rewards
        self.actions = actions

        self.num_states = np.prod(size)
        self.num_actions = len(GridWorld.POSSIBLE_ACTIONS)

    def _limit_coordinates(self, state):
        """Limits the coordinates if/after collision with grid wall

        Parameters
        ----------
        state : tuple (row, col)

        Returns
        -------
        state: tuple(row, col)

        """

        i, j = state
        if i < 0:
            i = 0
        elif i > self.height - 1:
            i = self.height - 1
        if j < 0:
            j = 0
        elif j > self.width - 1:
            j = self.width - 1
        return (i, j)

    def _new_state_reward(self, action, state):
        """Returns the coordinates of a resultant state and its rewards

        Parameters
        ----------
        action: char
            The character representing the action taken
        state: tuple (row, col)

        Returns
        -------
        state: tuple (row, col)
            The new state reached by taking the action
        reward: int
            The reward for the state

        """

        i, j = state
        if action == 'U':
            i, j = i - 1, j
        elif action == 'D':
            i, j = i + 1, j
        elif action == 'R':
            i, j = i, j + 1
        elif action == 'L':
            i, j = i, j - 1

        # make sure the new state is not out of grid
        new_state = self._limit_coordinates((i, j))

        return new_state, self.rewards.get(new_state)

    def transition(self, action, state, choose=False):
        """The stochastic transition model of the grid

        Parameters
        ----------
        action : char
            The character representing the action taken
        state : tuple (row, col)
            The current state from where transition is occuring
        choose : boolean
            If True: environment takes stochastic action and returns resultant
                     state and reward
            If False: environment returns a list of all possible actions with
                      corresponding reward and probabilties

        Returns
        -------
        If choose == True
            state : tuple (row, col)
            reward: int
        If choose == Flase
            A list with following tuple:
                prob: float
                    Probabilty with which environment selects the transition
                state: tuple (row, col)
                reward: int

        """

        def stochastic_transition(possible_actions, prob):
            if not choose:
                # create and return a list of all possible actions
                result = []
                for i, a in enumerate(possible_actions):
                    coord, reward = self._new_state_reward(a, state)
                    result.append((prob[i], coord, reward))
                return result
            else:
                # choose a random action with given probabilities
                a = np.random.choice(possible_actions, 1, p=prob)
                coord, reward = self._new_state_reward(a, state)
                return coord, reward

        if action == 'U':
            return stochastic_transition(['U', 'R', 'L'], [0.8, 0.1, 0.1])
        elif action == 'D':
            return stochastic_transition(['D', 'R', 'L'], [0.8, 0.1, 0.1])
        elif action == 'R':
            return stochastic_transition(['R', 'U', 'D'], [0.8, 0.1, 0.1])
        elif action == 'L':
            return stochastic_transition(['L', 'U', 'D'], [0.8, 0.1, 0.1])


def grid():
    """Utility function, returns 4x4 GridWorld object with rewards and actions
    """

    # dict with rewards for states of the grid
    rewards = {
        (0, 0): -1, (0, 1): -1, (0, 2): -1, (0, 3): -1,
        (1, 0): -1, (1, 1): -1, (1, 2): -1, (1, 3): -1,  # start state is 1x0
        (2, 0): -1, (2, 1): -70, (2, 2): -1, (2, 3): -1,  # bad state is 2x1
        (3, 0): -1, (3, 1): -1, (3, 2): -1, (3, 3): 100  # goal state is 3x3
    }

    # dict with actions allowed for the grid states
    actions = {
        (0, 0): ['R', 'D'], (0, 1): ['R', 'L', 'D'],
        (0, 2): ['R', 'L', 'D'], (0, 3): ['L', 'D'],
        (1, 0): ['R', 'U', 'D'], (1, 1): ['R', 'L', 'U', 'D'],
        (1, 2): ['R', 'L', 'U', 'D'], (1, 3): ['L', 'U', 'D'],
        (2, 0): ['R', 'U', 'D'], (2, 1): ['R', 'L', 'U', 'D'],
        (2, 2): ['R', 'L', 'U', 'D'], (2, 3): ['L', 'U', 'D'],
        (3, 0): ['R', 'U'], (3, 1): ['R', 'L', 'U'],
        (3, 2): ['R', 'L', 'U'], (3, 3): []
    }

    return GridWorld(size=(4, 4), rewards=rewards, actions=actions)


def print_grid(env, content_dict):
    """ Utility function that prints the grid environment with given content

    Parameters
    ---------
    env: GridWorld
    content_dict: dict {(row,col):object}
    """
    grid = np.arange(env.num_states, dtype=object).reshape(
        env.height, env.width)
    for coord, content in content_dict.items():
        grid[coord[0], coord[1]] = content
    print(grid)


def play_game(env, start, end, policy):
    """Utility function that follows given policy from start to end and
    returns a list of action, state and reward for each step

    Parameters
    ----------
    env : GridWorld
    start : (row, col)
    end : (row, col)
    policy : list of optimal actions for each state
        Policy list in the stats array returned by the algorithm

    Returns
    -------
    steps: list of [action,state,reward]
    """
    steps = []
    state = start
    while state != end:
        # get 1D index from state tuple
        state_idx = state[0] * env.height + state[1]
        action = policy[state_idx]
        # make a transition with choose=True
        new_state, reward = env.transition(action, state, choose=True)
        steps.append([action, list(state), reward])
        state = new_state
    # append the goal state for visualization
    steps.append(['G', list(end), 0])
    return steps
