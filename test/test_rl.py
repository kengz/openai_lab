import unittest
import pytest
from os import environ
environ['CI'] = environ.get('CI') or 'true'
import rl


class DQNTest(unittest.TestCase):

    def test_run_gym_tour(self):
        sys_vars = rl.main.run('dummy')
        assert(sys_vars['RENDER'] == False)
        # ensure it runs, and returns the sys_vars
        assert(isinstance(sys_vars, dict))
        assert(sys_vars['epi'] > 0)

    def test_run_q_table(self):
        sys_vars = rl.main.run('q_table')
        assert(sys_vars['RENDER'] == False)
        # ensure it runs, and returns the sys_vars
        assert(isinstance(sys_vars, dict))
        assert(sys_vars['epi'] > 0)

    def test_run_dqn(self):
        sys_vars = rl.main.run('dqn')
        assert(sys_vars['RENDER'] == False)
        # ensure it runs, and returns the sys_vars
        assert(isinstance(sys_vars, dict))
        assert(sys_vars['epi'] > 0)

    def test_run_dqn(self):
        sys_vars = rl.main.run('dqn')
        assert(sys_vars['RENDER'] == False)
        # ensure it runs, and returns the sys_vars
        assert(isinstance(sys_vars, dict))
        assert(sys_vars['epi'] > 0)

    def test_run_double_dqn(self):
        sys_vars = rl.main.run('double_dqn')
        assert(sys_vars['RENDER'] == False)
        # ensure it runs, and returns the sys_vars
        assert(isinstance(sys_vars, dict))
        assert(sys_vars['epi'] > 0)
