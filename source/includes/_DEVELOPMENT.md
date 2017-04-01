# <a name="development"></a>Development

**(THIS SECTION IS UNDER CONSTRUCTION)**

(pending writeup)

This is still under active development, and documentation is sparse. The main code lives inside `rl/`.

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

Each run is an `experiment` that runs multiple `Trial`s (not restricted to the same `experiment_id` for future cross-training). Each `Trial` runs multiple (by flag `-t`) `Session`s, so an `trial` is a `sess_grid`.

Each trial collects the data from its sessions into `trial_data`, which is saved to a JSON and as many plots as there are sessions. On the higher level, `experiment` analyses the aggregate `trial_data` to produce a best-sorted CSV and graphs of the variables (what's changed across experiments) vs outputs.

## problem

simple, the split, how to add. each JSON key is?