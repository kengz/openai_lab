from rl.experiment import run

if __name__ == '__main__':
    run('dev_dqn', times=2, param_selection=True)
    # run('dqn', times=2, param_selection=False)
    # run('lunar_dqn', times=1, param_selection=False)
    # run('DevCartPole-v0_DQN_HighLowMemoryWithForgetting_BoltzmannPolicy_NoPreProcessor_2017-01-21_191023_e0', plot_only=True)
    # run('lunar_dqn', times=3, param_selection=True, line_search=True)
