from environment import grid, print_grid
from value_iteration import value_iteration
from q_learning import q_learning

from flask import Flask,request,render_template
import json

app = Flask(__name__)
env = grid()

@app.route('/valueiteration')
def dp():
    policy, V, stats = value_iteration(env,(1,0),(3,3))
    print_grid(env, policy)
    print_grid(env, V)
    return render_template('value_iteration.html',data=json.dumps(stats))

@app.route('/qlearning',methods=['GET'])
def td():
    episodes = int(request.args.get('episodes',300))
    epsilon = float(request.args.get('epsilon',0.05))
    alpha = float(request.args.get('alpha',0.5))
    policy, V, stats = q_learning(env,episodes,epsilon,alpha,(1,0),(3,3))
    print_grid(env, policy)
    print_grid(env, V)
    return render_template('q_learning.html',data = json.dumps(stats[10:]))

if __name__ == '__main__':
    app.run()



