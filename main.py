from rl.experiment import run

if __name__ == '__main__':
    # ['dummy', 'q_table', 'lunar_double_dqn',
    #     'mountain_double_dqn', 'lunar_dqn', 'double_dqn', 'dqn']
    run('dqn', times=2, param_selection=False)
    # run('lunar_dqn', times=5, param_selection=True, line_search=True)
