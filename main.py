from rl.experiment import game_specs, run

if __name__ == '__main__':
    # print(list(game_specs.keys()))
    # ['dummy', 'q_table', 'lunar_double_dqn',
    #     'mountain_double_dqn', 'lunar_dqn', 'double_dqn', 'dqn']
    run('dqn', times=1, param_selection=False)
    # run('lunar_dqn', times=5, param_selection=True, line_search=True)
