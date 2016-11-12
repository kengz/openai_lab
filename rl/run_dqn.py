from runner import Runner
from agent.dqn import DQN


if __name__ == '__main__':
    dqn_runner = Runner(DQN,
                        problem='CartPole-v0',
                        param={'e_anneal_steps': 5000,
                               'learning_rate': 0.01,
                               'gamma': 0.99})
    dqn_runner.run_session()
