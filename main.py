from environment import grid, print_grid
from value_iteration import value_iteration

env = grid()
policy , V = value_iteration(env)
print_grid(env,policy)
print_grid(env,V)