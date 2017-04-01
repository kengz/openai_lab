# <a name="development"></a>Development


**This section is still under construction**.

For agent-specific development, see [Agents](#agents). This section details the general, non-agent development guideline.

The design of the code is clean enough to simply infer how things work by example.

- `data/`: data folders grouped per experiment, each of which contains all the graphs per trial sessions, JSON data file per trial, and csv metrics dataframe per run of multiple trials
- `rl/agent/`: custom agents. Refer to `base_agent.py` and `dqn.py` to build your own
- `rl/hyperoptimizer/`: Hyperparameter optimizers for the Experiments
- `rl/memory/`: RL agent memory classes
- `rl/optimizer/`: RL agent NN optimizer classes
- `rl/policy/`: RL agent policy classes
- `rl/preprocessor/`: RL agent preprocessor (state and memory) classes
- `rl/spec/`: specify new problems and experiment_specs to run experiments for.
- `rl/analytics.py`: the data analytics module for output experiment data
- `rl/experiment.py`: the main high level experiment logic
- `rl/util.py`: Generic util


## problem

simple, the split, how to add. each JSON key is?
