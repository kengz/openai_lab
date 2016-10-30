import unittest
import pytest
from os import environ
environ['CI'] = environ.get('CI') or 'true'
import rl


class DQNTest(unittest.TestCase):

    def test_run_gym_tour(self):
        param = {'e_anneal_steps': 10000,
                 'learning_rate': 0.01,
                 'n_epoch': 1,
                 'gamma': 0.99}
        assert(rl.run_gym_tour.sys_vars['RENDER'] == False)
        sys_vars = rl.run_gym_tour.run_session(param)
        # ensure it runs, and returns the sys_vars
        assert(isinstance(sys_vars, dict))
        assert(sys_vars['epi'] > 0)

    def test_run_dqn(self):
        param = {'e_anneal_steps': 10000,
                 'learning_rate': 0.01,
                 'n_epoch': 1,
                 'gamma': 0.99}
        assert(rl.run_dqn.sys_vars['RENDER'] == False)
        sys_vars = rl.run_dqn.run_session(param)
        # ensure it runs, and returns the sys_vars
        assert(isinstance(sys_vars, dict))
        assert(sys_vars['epi'] > 0)
