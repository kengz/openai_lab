from rl.session import sess_specs, run

if __name__ == '__main__':
    # print(list(sess_specs.keys()))
    # ['dummy', 'q_table', 'lunar_double_dqn', 'mountain_double_dqn', 'lunar_dqn', 'double_dqn', 'dqn']
    run('mountain_double_dqn')
