from rl.experiment import run
from util import args

if __name__ == '__main__':
    run('dqn', times=5, param_selection=True, line_search=False)
    run('lunar_dqn', times=5, param_selection=True, line_search=False)
    run('sarsa_dqn', times=5, param_selection=True, line_search=False)
    run('sarsa_exp', times=5, param_selection=True, line_search=False)
