import unittest
import pytest
from os import environ
environ['CI'] = environ.get('CI') or 'true'
from rl.experiment import run


class DQNTest(unittest.TestCase):

    def test_run_gym_tour(self):
        metrics_df = run('dummy')

    def test_run_q_table(self):
        metrics_df = run('q_table')

    def test_run_dqn(self):
        metrics_df = run('dqn')

    def test_run_double_dqn(self):
        metrics_df = run('double_dqn')

    def test_run_mountain_double_dqn(self):
        metrics_df = run('mountain_double_dqn')

    # def test_run_all(self):
    #     for x in game_specs.keys():
    #         metrics_df = run(x)
