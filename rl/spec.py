from rl.agent import *
from rl.memory import *
from rl.policy import *

# Dict of specs runnable on a Session
# specify the Class constructors for Agent, Memory, Policy
# specify any of their parameters under the unified 'param' key
# specify param_range for hyper param selection (if needed)
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
        # 'Memory': LinearMemoryWithForgetting,
        'Memory': RankedMemory,
        'Policy': BoltzmannPolicy,
        'param': {
            'train_per_n_new_exp': 1,
            'learning_rate': 0.02,
            'gamma': 0.99,
            'hidden_layers_shape': [4],
            'hidden_layers_activation': 'sigmoid',
            'exploration_anneal_episodes': 10,
            'state_preprocessing' : 'none',
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
            'state_preprocessing' : 'none',
        }
    },
    # Mountain dqn params don't work yet
    'mountain_dqn': {
        'problem': 'MountainCar-v0',
        'Agent': lunar_dqn.LunarDQN,
        'Memory': LinearMemoryWithForgetting,
        'Policy': EpsilonGreedyPolicy,
        'param': {
            'train_per_n_new_exp': 3,
            'learning_rate': 0.001,
            'batch_size': 32,
            'gamma': 0.98,
            'hidden_layers_shape': [200, 100],
            'hidden_layers_activation': 'relu',
            'exploration_anneal_episodes': 100,
            'state_preprocessing' : 'none',
        },
        'param_range': {
            'learning_rate': [0.001, 0.01],
            'hidden_layers_shape': [[200, 100], [200, 100, 50]],
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
            'state_preprocessing' : 'none',
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
            'gamma': 0.99,
            'hidden_layers_shape': [200, 100],
            'hidden_layers_activation': 'relu',
            'output_layer_activation': 'linear',
            'exploration_anneal_episodes': 100,
            'epi_change_learning_rate' : 350,
            'state_preprocessing' : 'none',
        },
        'param_range': {
            'train_per_n_new_exp': [1, 2, 3, 4, 5, 8, 10, 15],
            'gamma' : [0.95, 0.96, 0.97, 0.98, 0.99],
            'exploration_anneal_episodes': [100, 200, 300, 400, 500, 600],
            'learning_rate': [0.01, 0.005, 0.001, 0.0005, 0.0001],
            'hidden_layers_shape': [[100], [200], [300], [400], [500], 
                                    [200, 100], [300, 100], [300, 150], 
                                    [400, 100], [400, 200]],
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
            'state_preprocessing' : 'none',
        }
    },
    'air_raid_dqn': {
        'problem': 'AirRaid-v0',
        'Agent': atari_conv_dqn.ConvDQN,
        'Memory': LongLinearMemoryWithForgetting,
        'Policy': EpsilonGreedyPolicy,
        'param': {
            'train_per_n_new_exp': 4,
            'learning_rate': 0.001,
            'batch_size': 32,
            'gamma': 0.99,
            'hidden_layers_shape': [[16, 8, 8, (4, 4)], [32, 4, 4, (2, 2)]],
            'hidden_layers_activation': 'relu',
            'exploration_anneal_episodes': 1000000,
            'epi_change_learning_rate' : 1000000,
        },
        'param_range': {
            'learning_rate': [0.001, 0.01],
            'hidden_layers_shape': [[200, 100], [300, 200], [200, 100, 50]],
        }
    },
    'breakout_dqn': {
        'problem': 'Breakout-v0',
        'Agent': atari_conv_dqn.ConvDQN,
        'Memory': LongLinearMemoryWithForgetting,
        'Policy': EpsilonGreedyPolicy,
        'param': {
            'train_per_n_new_exp': 4,
            'learning_rate': 0.001,
            'batch_size': 32,
            'gamma': 0.99,
            'hidden_layers_shape': [[16, 8, 8, (4, 4)], [32, 4, 4, (2, 2)]],
            'hidden_layers_activation': 'relu',
            'exploration_anneal_episodes': 1000000,
            'epi_change_learning_rate' : 1000000,
        },
        'param_range': {
            'learning_rate': [0.001, 0.01],
            'hidden_layers_shape': [[200, 100], [300, 200], [200, 100, 50]],
        }
    }
}
