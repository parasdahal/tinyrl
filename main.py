from environment import grid
from value_iteration import *

env = grid()
policy , V = value_iteration(env)
print_solution(env,policy)
print_solution(env,V)