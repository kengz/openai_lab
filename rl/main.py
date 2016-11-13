from runner import Runner
from agent import *

sess_specs = {
    'dummy': {
        'Agent': dummy.Dummy,
        'problem': 'CartPole-v0',
        'param': {}
    },
    'q_table': {
        'Agent': q_table.QTable,
        'problem': 'CartPole-v0',
        'param': {'e_anneal_steps': 5000,
                  'learning_rate': 0.01,
                  'gamma': 0.99}
    },
    'dqn': {
        'Agent': dqn.DQN,
        'problem': 'CartPole-v0',
        'param': {'e_anneal_steps': 5000,
                  'learning_rate': 0.01,
                  'gamma': 0.99}
    },
    'double_dqn': {
        'Agent': double_dqn.DoubleDQN,
        'problem': 'CartPole-v0',
        'param': {'e_anneal_steps': 2500,
                  'learning_rate': 0.01,
                  'batch_size': 32,
                  'gamma': 0.97}
    },
    'mountain_double_dqn': {
        'Agent': mountain_double_dqn.MountainDoubleDQN,
        'problem': 'MountainCar-v0',
        'param': {'e_anneal_steps': 10000,
                  'learning_rate': 0.01,
                  'batch_size': 128,
                  'gamma': 0.97}
    },
    'lunar_dqn': {
        'Agent': lunar_dqn.LunarDQN,
        'problem': 'LunarLander-v2',
        'param': {'e_anneal_steps': 150000,
                  'learning_rate': 0.01,
                  'batch_size': 128,
                  'gamma': 0.99}
    },
    'lunar_double_dqn': {
        'Agent': lunar_double_dqn.LunarDoubleDQN,
        'problem': 'LunarLander-v2',
        'param': {'e_anneal_steps': 250000,
                  'learning_rate': 0.01,
                  'batch_size': 128,
                  'gamma': 0.99}
    }
}


def run(sess_name):
    sess_spec = sess_specs.get(sess_name)
    sess_runner = Runner(sess_spec['Agent'],
                         problem=sess_spec['problem'],
                         param=sess_spec['param'])
    return sess_runner.run_session()

if __name__ == '__main__':
    run('mountain_double_dqn')
