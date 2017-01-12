from rl.session import game_specs, run

if __name__ == '__main__':
    # print(list(game_specs.keys()))
    # ['dummy', 'q_table', 'lunar_double_dqn',
    #     'mountain_double_dqn', 'lunar_dqn', 'double_dqn', 'dqn']
    # run('dqn', run_param_selection=False, times=1)
    run('lunar_dqn', run_param_selection=False, times=1, line_search=True)
