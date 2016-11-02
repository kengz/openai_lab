import unittest
import pytest
from os import environ
environ['CI'] = environ.get('CI') or 'true'
import rl


class DQNTest(unittest.TestCase):

    def test_run_gym_tour(self):
        problem = 'CartPole-v0'
        param = {'e_anneal_steps': 10000,
                 'learning_rate': 0.01,
                 'n_epoch': 1,
                 'gamma': 0.99}
        sys_vars = rl.run_gym_tour.run_session(problem, param)
        assert(sys_vars['RENDER'] == False)
        # ensure it runs, and returns the sys_vars
        assert(isinstance(sys_vars, dict))
        assert(sys_vars['epi'] > 0)

    def test_run_tabular_q(self):
        problem = 'CartPole-v0'
        param = {'e_anneal_steps': 10000,
                 'learning_rate': 0.01,
                 'gamma': 0.99}
        sys_vars = rl.run_tabular_q.run_session(problem, param)
        assert(sys_vars['RENDER'] == False)
        # ensure it runs, and returns the sys_vars
        assert(isinstance(sys_vars, dict))
        assert(sys_vars['epi'] > 0)

    def test_run_dqn(self):
        problem = 'CartPole-v0'
        param = {'e_anneal_steps': 10000,
                 'learning_rate': 0.01,
                 'n_epoch': 1,
                 'gamma': 0.99}
        sys_vars = rl.run_dqn.run_session(problem, param)
        assert(sys_vars['RENDER'] == False)
        # ensure it runs, and returns the sys_vars
        assert(isinstance(sys_vars, dict))
        assert(sys_vars['epi'] > 0)
