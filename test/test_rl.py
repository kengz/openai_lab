import unittest
import pytest
from os import environ
environ['CI'] = environ.get('CI') or 'true'
from rl.session import sess_specs, run


class DQNTest(unittest.TestCase):

    def test_run_gym_tour(self):
        sys_vars = run('dummy')
        assert(sys_vars['RENDER'] == False)
        # ensure it runs, and returns the sys_vars
        assert(isinstance(sys_vars, dict))
        assert(sys_vars['epi'] > 0)

    def test_run_q_table(self):
        sys_vars = run('q_table')
        assert(sys_vars['RENDER'] == False)
        # ensure it runs, and returns the sys_vars
        assert(isinstance(sys_vars, dict))
        assert(sys_vars['epi'] > 0)

    def test_run_dqn(self):
        sys_vars = run('dqn')
        assert(sys_vars['RENDER'] == False)
        # ensure it runs, and returns the sys_vars
        assert(isinstance(sys_vars, dict))
        assert(sys_vars['epi'] > 0)

    def test_run_double_dqn(self):
        sys_vars = run('double_dqn')
        assert(sys_vars['RENDER'] == False)
        # ensure it runs, and returns the sys_vars
        assert(isinstance(sys_vars, dict))
        assert(sys_vars['epi'] > 0)

    # def test_run_all(self):
    #     for x in sess_specs.keys():
    #         sys_vars = run('dummy')
    #         assert(sys_vars['RENDER'] == False)
    #         # ensure it runs, and returns the sys_vars
    #         assert(isinstance(sys_vars, dict))
    #         assert(sys_vars['epi'] > 0)
