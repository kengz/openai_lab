from rl.agent import *
from rl.memory import *
from rl.policy import *

# Dict of specs runnable on a Session
game_specs = {
    'dummy': {
        'problem': 'CartPole-v0',
        'Agent': dummy.Dummy,
        'Memory': LinearMemory,
        'Policy': EpsilonGreedyPolicy,
        'param': {}
    },
    'q_table': {
        'problem': 'CartPole-v0',
        'Agent': q_table.QTable,
        'Memory': LinearMemory,
        'Policy': EpsilonGreedyPolicy,
        'param': {
            'learning_rate': 0.01,
            'gamma': 0.99,
            'exploration_anneal_episodes': 200,
        }
    },
    'dqn': {
        'problem': 'CartPole-v0',
        'Agent': dqn.DQN,
        'Memory': LinearMemoryWithForgetting,
        'Policy': BoltzmannPolicy,
        'param': {
            'train_per_n_new_exp': 1,
            'learning_rate': 0.02,
            'gamma': 0.99,
            'hidden_layers_shape': [4],
            'hidden_layers_activation': 'sigmoid',
            'exploration_anneal_episodes': 10,
        },
        'param_range': {
            'learning_rate': [0.01, 0.05, 0.1],
            'gamma': [0.99],
            'exploration_anneal_episodes': [50, 100],
        }
    },
    'double_dqn': {
        'problem': 'CartPole-v0',
        'Agent': double_dqn.DoubleDQN,
        'Memory': LinearMemory,
        'Policy': EpsilonGreedyPolicy,
        'param': {
            'train_per_n_new_exp': 1,
            'learning_rate': 0.01,
            'batch_size': 32,
            'gamma': 0.99,
            'hidden_layers_shape': [4],
            'hidden_layers_activation': 'sigmoid',
            'exploration_anneal_episodes': 180,
        }
    },
    'mountain_double_dqn': {
        'problem': 'MountainCar-v0',
        'Agent': mountain_double_dqn.MountainDoubleDQN,
        'Memory': LinearMemory,
        'Policy': EpsilonGreedyPolicy,
        'param': {
            'train_per_n_new_exp': 1,
            'learning_rate': 0.01,
            'batch_size': 128,
            'gamma': 0.99,
            'hidden_layers_shape': [8, 8],
            'hidden_layers_activation': 'sigmoid',
            'exploration_anneal_episodes': 300,
        }
    },
    'lunar_dqn': {
        'problem': 'LunarLander-v2',
        'Agent': lunar_dqn.LunarDQN,
        'Memory': LinearMemoryWithForgetting,
        'Policy': EpsilonGreedyPolicy,
        'param': {
            'train_per_n_new_exp': 4,
            'learning_rate': 0.001,
            'batch_size': 32,
            'gamma': 0.98,
            'hidden_layers_shape': [200, 100],
            'hidden_layers_activation': 'relu',
            'exploration_anneal_episodes': 300,
        },
        'param_range': {
            'learning_rate': [0.001, 0.01],
            'hidden_layers_shape': [[200, 100], [200, 100, 50]],
        }
    },
    'lunar_double_dqn': {
        'problem': 'LunarLander-v2',
        'Agent': lunar_double_dqn.LunarDoubleDQN,
        'Memory': LinearMemory,
        'Policy': EpsilonGreedyPolicy,
        'param': {
            'train_per_n_new_exp': 1,
            'learning_rate': 0.01,
            'batch_size': 64,
            'gamma': 0.99,
            'hidden_layers_shape': [200, 100],
            'hidden_layers_activation': 'relu',
            'exploration_anneal_episodes': 500,
        }
    }
}
