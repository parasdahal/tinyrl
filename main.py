from environment import grid, print_grid
from value_iteration import value_iteration
from q_learning import q_learning
import matplotlib.pyplot as plt

env = grid()
policy, V = value_iteration(env)
print_grid(env, policy)
print_grid(env, V)
policy, V, average_q = q_learning(env,1000,0.05,0.5,(1,0),(3,3))
print_grid(env, policy)
print_grid(env, V)
plt.plot(average_q)
plt.show()