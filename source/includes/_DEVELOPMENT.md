# <a name="development"></a>Development

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

## Agent
## HyperOptimizer

use which first: 

### LineSearch
### GridSearch
### RandomSearch


### HyperOptimizer Roadmap

These are the future hyperparameter optimization algorithms we'd like to implement standalone in the Lab. The implementations for them currently exists, but they're too bloated, and their engineering aspects are not ideal for the Lab.

- [TPE](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf)/[hyperopt](https://github.com/hyperopt/hyperopt)
- [Bayesian Optimizer (Spearmint)](https://github.com/HIPS/Spearmint)
- [SMAC](http://www.cs.ubc.ca/labs/beta/Projects/SMAC/#software)


### Implementation Guideline

All implemented hyperoptimizers shall extend the base `HyperOptimizer` class in `rl/hyperoptimizer/base_hyperoptimizer.py` and follow its design for compatibility. Below we show this design to be general theoretically and practically. Moreover, do not use bloated dependencies.

**Theoretical design:**

A hyperoptimizer is a function `h` that takes:

- a trial (objective) function `Trial`
- a parameter space `P` (implemented in `experiment_spec`)

and runs the algorithm:

1. search the next `p` in `P` using its internal search algorithm, add to its internal `param_search_list`.
2. run a (slow) function `Trial(p) = fitness_score` (inside trial data)
3. update search using the feedback `fitness_score`
4. repeat until max steps or fitness condition met

Note that the search space `P` is a tensor space product of `m` bounded real spaces `R` and `n` bounded discrete spaces `N`. The search path in `param_search_list` must also be well-ordered to ensure resumability.


**Implementation requirements:**

1. we want order-preserving and persistence in search for the ability to resume/reproduce an experiment.
2. the search algorithm may have its own internal memory/belief to facilitate search.
3. the `Trial` function shall be treated as a blackbox `Trial(p) = fitness_score` with input/output `(p, fitness_score)` for the generality of implementation/


**Specification of search space:**

1\. for real variable, specify a distribution (an interval is just a uniformly distributed space) in the `experiment_spec.param_range`. Example:

```json
"lr": {
  "min": 0.0005,
  "max": 0.05
}
```

2\. for discrete variable, specify a list of the values to search over (since it is finite anyway) in the `experiment_spec.param_range`. This will automatically be sorted when read into `HyperOptimizer` to ensure ordering. Example:


```json
"lr": [0.01, 0.02, 0.05, 0.1, 0.2]
```


The hyperopt implementation shall be able to take these 2 types of param_range specs and construct its search space.

Note that whether a variable is real or discrete can be up to the user; some variable such as `lr` can be sampled from interval `0.001 to 0.1` or human-specified options `[0.01, 0.02, 0.05, 0.1, 0.2]`. One way may be more efficient than the other depending on the search algorithm.

The experiment will run it as:

```python
# specify which hyperoptimizer class to use in spec for bookkeeping
Hopt = get_module(GREF, experiment_spec['HyperOptimizer'])
hopt = Hopt(Trial, **experiment_kwargs)
experiment_data = hopt.run()
```


## Memory
## Optimizer
## Policy
## PreProcessor

