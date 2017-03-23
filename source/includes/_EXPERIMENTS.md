# <a name="experiments"></a>Experiments

The experimental framework design and terminology should be familiar, since it's borrowed from experimental science. The Lab runs **experiments** and produces data for [analysis](#analysis).


## Definition

An **experiment** runs separate **trials** by varying parameters. Each **trial** runs multiple **sessions** for averaging the results.

An experiment consists of:

- an **environment** (problem) from [OpenAI Gym](https://gym.openai.com/envs)
- an **agent** to solve the environment.

<aside class="notice">
An experiment runs the variations of agent by changing its parameters (experiment variables) while holding others constants (control), and measure the fitness_score (outcome) to solve the environment.
</aside>


## Specification

An experiment is specified by an `experiment_spec` in `rl/spec/*_experiment_specs.json`.

```json
{
  "dqn": {
    "problem": "CartPole-v0",
    "Agent": "DQN",
    "HyperOptimizer": "GridSearch",
    "Memory": "LinearMemoryWithForgetting",
    "Optimizer": "AdamOptimizer",
    "Policy": "BoltzmannPolicy",
    "PreProcessor": "NoPreProcessor",
    "param": {
      "lr": 0.02,
      "gamma": 0.99,
      "hidden_layers": [64],
      "hidden_layers_activation": "sigmoid",
      "exploration_anneal_episodes": 10
    },
    "param_range": {
      "lr": [0.001, 0.005, 0.01, 0.02],
      "gamma": [0.95, 0.97, 0.99, 0.999],
      "hidden_layers": [
        [16],
        [32],
        [64],
        [16, 8],
        [32, 16]
      ]
    }
  }
}
```

It consists of:

- `experiment_name`: the key of the JSON. e.g. `dqn`
- `problem`: name of the environment. e.g. `CartPole-v0`
- **agent**: and its components in `rl/`, specified by the class name
    - `Agent`: the main agent class, typically with the algorithm like `DQN, DDPG`
    - `HyperOptimizer`: hyperparameter optimization algorithms used to vary the agent parameters and run trials with them
    - `Memory`: the memory module of the agent, to specify how agent control or access memory
    - `Optimizer`: the neural network optimizer of Agent
    - `Policy`: the externalized policy of Agent
    - `PreProcessor`: for the environment states. Useful for Atari with images.
- `param`: the default parameter values used (control variables)
- `param_range`: the hyperparameter space ranges to search through by `HyperOptimimzer` (experiment variables).


## Process Hierarchy

How `experiments > trials > sessions` are organized and ran.

When the Lab runs an **experiment** with `experiment_name` (e.g. `dqn`):

- it creates a timestamped `experiment_id` (`dqn-2017_03_19_004714`)
- the **experiment** runs multiple trials over the hyperparameter space
    - the trials are ordered for resumability (in case machine dies)
    - each trial has `trial_id` (`dqn-2017_03_19_004714_t0`), tied to a unique set of param values
    - a **trial** runs multiple sessions
        - each **session** has `session_id` (`dqn-2017_03_19_004714_t0_s0`)
        - a session runs the environment-agent, produces graphs and `session_data` i.e. `sys_vars`
        - the session saves its graph to `<session_id>.png`
        - the session returns `sys_vars` to its trial
    - the trial gathers all the `sys_vars`, run some averaging analytics, then compose all that into `trial_data`
    - the trial returns the `trial_data` and saves it to `<trial_id>.json`
- the experiment composes all `trial_data` into a `experiment_data`
- it runs analytics to produce graphs `<experiment_id>_analysis.png, <experiment_id>_correlation.png`
- it compute the `fitness_score` for each trial, rank them by best-first, then save the data grid to `<experiment_id>_analysis_data.csv`
- experiment ends


## Evaluation Metrics

OpenAI Lab exists to address 2 major problems in RL, and WildML's Denny sums them up best in his post [Engineering Is The Bottleneck In (Deep Learning) Research](http://blog.dennybritz.com/2017/01/17/engineering-is-the-bottleneck-in-deep-learning-research/). They are:

**1. the difficulty of building upon other’s work**

As the Lab grows, we hope that engineers and researchers can experiment with an idea fast by building on top of our existing components.

**2. the lack of rigor in comparisons**

Multiple experiments running in the Lab will produce the same analytics and the evaluation metrics. This will allow us to compare algorithms and problems meaningfully, and that is the point of the Lab's [Solution Matrix](#solution-matrix).

We now describe the evaluation metrics for **problems** and **algorithms**.

### Problem Evaluation Metrics

problem x {algorithms} ~ solutions

fitness score on 4 parts:
stability and reproducibility (solve ratio),
speed (min episodes),
potential (max reward),
square for granularity

```
mean_rewards_per_epi * (1+solved_ratio_of_sessions)**2

ideal_mean_rewards_per_epi = mean_rewards / (epi/solved_epi_speedup)
ideal_solved_ratio = 1
ideal_fitness_score = fitness_score(
    ideal_mean_rewards_per_epi, ideal_solved_ratio)
return ideal_fitness_score
```

### Algorithm Evaluation Metrics

algorithm x {problems} ~ cross-solutions


## Lab Demo

Given the framework explained above, here's a quick demo. Suppose we aim to solve the CartPole-v0 problem with the plain DQN agent.  Suppose again for this experiment, we implement a new agent component, namely a `Boltzmann` policy, and try to find the best parameter sets for this new agent.

### Specify Experiment

The example below is fully specified in `rl/spec/classic_experiment_specs.json` under `dqn`:

```json
{
  "dqn": {
    "problem": "CartPole-v0",
    "Agent": "DQN",
    "HyperOptimizer": "GridSearch",
    "Memory": "LinearMemoryWithForgetting",
    "Optimizer": "AdamOptimizer",
    "Policy": "BoltzmannPolicy",
    "PreProcessor": "NoPreProcessor",
    "param": {
      "lr": 0.02,
      "gamma": 0.99,
      "hidden_layers": [64],
      "hidden_layers_activation": "sigmoid",
      "exploration_anneal_episodes": 10
    },
    "param_range": {
      "lr": [0.001, 0.005, 0.01, 0.02],
      "gamma": [0.95, 0.97, 0.99, 0.999],
      "hidden_layers": [
        [16],
        [32],
        [64],
        [16, 8],
        [32, 16]
      ]
    }
  }
}
```

Specifically of interests, we have specified the variables:

- *experiment_name*: `dqn`
- *problem*: [CartPole-v0](https://gym.openai.com/envs/CartPole-v0)
- *variable agent component*: `Boltzmann` policy
- *control agent variables*:
    - `DQN` agent
    - `LinearMemoryWithForgetting`
    - `AdamOptimizer`
    - `NoPreProcessor`
- *hyperparameter space*: the `"param_range"` JSON
- *hyperparameter optimizer*: `GridSearch`

Given `GridSearch HyperOptimizer`, this **experiment** will try all the discrete combinations of the `param_range`, which makes for `4x4x5=80` trials. Each **trial** will run a max of 5 **sessions** (terminate on 2 if fail to solve). Overall, this experiments will run at most `80 x 5 = 400` sessions, then produce `experiment_data` and the analytics.


### Lab Workflow

The example workflow to setup this experiment is as follow:

1. Add the new theorized component `Boltzmann` in `rl/policy/boltzmann.py`
2. Specify `dqn` experiment spec in `rl/spec/classic_experiment_spec.json` to include this new variable, reuse the other existing RL components, and specify the param range.
3. Add this experiment to the lab queue in `config/production.json`
4. Run experiment with `grunt -prod`
5. Analyze the graphs and data


Next, see the example lab results, and how to [analyze the data and graphs](#analysis).
