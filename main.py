from rl.experiment import run

if __name__ == '__main__':
    # ['dummy', 'q_table', 'lunar_double_dqn',
    #     'mountain_double_dqn', 'lunar_dqn', 'double_dqn', 'dqn']
    run('dqn', times=2, param_selection=False)
    # run('CartPole-v0_DQN_LinearMemoryWithForgetting_BoltzmannPolicy_2017-01-14_144625', plot_only=True)
    # run('lunar_dqn', times=5, param_selection=True, line_search=True)
