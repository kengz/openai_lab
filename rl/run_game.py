from runner import Runner
from agent import *

if __name__ == '__main__':
    # q_table_runner = Runner(q_table.QTable,
    #                     problem='CartPole-v0',
    #                     param={'e_anneal_steps': 5000,
    #                            'learning_rate': 0.01,
    #                            'gamma': 0.99})
    # q_table_runner.run_session()

    # dqn_runner = Runner(dqn.DQN,
    #                     problem='CartPole-v0',
    #                     param={'e_anneal_steps': 5000,
    #                            'learning_rate': 0.01,
    #                            'gamma': 0.99})
    # dqn_runner.run_session()

    # double_dqn_runner = Runner(double_dqn.DoubleDQN,
    #                            problem='CartPole-v0',
    #                            param={'e_anneal_steps': 2500,
    #                                   'learning_rate': 0.01,
    #                                   'batch_size': 32,
    #                                   'gamma': 0.97})
    # double_dqn_runner.run_session()

    # lunar_dqn_runner = Runner(lunar_dqn.LunarDQN,
    #                           problem='LunarLander-v2',
    #                           param={'e_anneal_steps': 150000,
    #                                  'learning_rate': 0.01,
    #                                  'batch_size': 128,
    #                                  'gamma': 0.99})
    # lunar_dqn_runner.run_session()

    lunar_double_dqn_runner = Runner(lunar_double_dqn.LunarDoubleDQN,
                                     problem='LunarLander-v2',
                                     param={'e_anneal_steps': 150000,
                                            'learning_rate': 0.01,
                                            'batch_size': 128,
                                            'gamma': 0.99})
    lunar_double_dqn_runner.run_session()
