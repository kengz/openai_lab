# OpenAI Lab [![CircleCI](https://circleci.com/gh/kengz/openai_lab.svg?style=shield)](https://circleci.com/gh/kengz/openai_lab) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/a0e6bbbb6c4845ccaab2db9aecfecbb0)](https://www.codacy.com/app/kengzwl/openai_lab?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=kengz/openai_lab&amp;utm_campaign=Badge_Grade) [![Codacy Badge](https://api.codacy.com/project/badge/Coverage/9e55f845b10b4b51b213620bfb98e4b3)](https://www.codacy.com/app/kengzwl/openai_lab?utm_source=github.com&utm_medium=referral&utm_content=kengz/openai_lab&utm_campaign=Badge_Coverage) [![GitHub forks](https://img.shields.io/github/forks/kengz/openai_lab.svg?style=social&label=Fork)](https://github.com/kengz/openai_lab) [![GitHub stars](https://img.shields.io/github/stars/kengz/openai_lab.svg?style=social&label=Star)](https://github.com/kengz/openai_lab)

---

<p align="center"><b><a href="http://kengz.me/openai_lab">OpenAI Lab Documentation</a></b></p>

---


_An experimentation system for Reinforcement Learning using OpenAI and Keras._

The _OpenAI Lab_ is created to do Reinforcement Learning (RL) like science - _theorize, experiment_. It provides an easy to use interface to [OpenAI Gym](https://gym.openai.com/) and [Keras](https://keras.io/), combined with an automated experimental and analytics framework.

While these are powerful tools, they take a lot to get running. Of many implementations we saw which solve OpenAI gym environments, many had to rewrite the same basic components instead of just the new components being researched.

To address this, the Lab does three things.

1. Handles the basic RL environment and algorithm setups.
2. Provides a standard, extensible platform with reusable components for developing deep reinforcement learning algorithms.
3. Provides a rigorous experimentation system with logs, plots and analytics for testing new RL algorithms. Experimental settings are logged in a standardized format so that solutions can be reproduced by anyone using the Lab.

With OpenAI Lab, we could focus on researching the essential elements of reinforcement learning such as the algorithm, policy, memory (experience replay), and parameter tuning to solve the OpenAI environments. We could also test our hypotheses more reliably.

<img alt="Timelapse of OpenAI Lab" src="http://kengz.me/openai_lab/images/lab_demo_dqn.gif" />
_Timelapse of OpenAI Lab, solving CartPole-v0._


## Lab Demo

Each experiment involves:
- a problem - an [OpenAI Gym environment](https://gym.openai.com/envs)
- a RL agent with modular components `agent, memory, optimizer, policy, preprocessor`, each of which is an experimental variable.

We specify input parameters for the experimental variable, run the experiment, record and analyze the data, conclude if the agent solves the problem with high rewards.

### Specify Experiment

Each experiment involves:
- a problem - an [OpenAI Gym environment](https://gym.openai.com/envs)
- a RL agent with modular components `agent, memory, optimizer, policy, preprocessor`, each of which is an experimental variable.

We specify input parameters for the experimental variable, run the experiment, record and analyze the data, conclude if the agent solves the problem with high rewards.

### Specify Experiment

The example below is fully specified in `rl/asset/classic_experiment_specs.json` under `dqn`:

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
      "lr": 0.01,
      "decay": 0.0,
      "gamma": 0.99,
      "hidden_layers": [32],
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

- *experiment*: `dqn`
- *problem*: [CartPole-v0](https://gym.openai.com/envs/CartPole-v0)
- *variable agent component*: `Boltzmann` policy
- *control agent variables*:
    - `DQN` agent
    - `LinearMemoryWithForgetting`
    - `AdamOptimizer`
    - `NoPreProcessor`
- *parameter variables values*: the `"param_range"` JSON

An **experiment** will run a trial for each combination of `param` values; each **trial** will run for multiple repeated **sessions**. For `dqn`, there are `4x4x5=80` param combinations (trials), and up to `5` repeated sessions per trial. Overall, this experiment will run at most `80 x 5 = 400` sessions.


### Lab Workflow

The workflow to setup this experiment is as follow:

1. Add the new theorized component `Boltzmann` in `rl/policy/boltzmann.py`
2. Specify `dqn` experiment spec in `experiment_spec.json` to include this new variable,  reuse the other existing RL components, and specify the param range.
3. Add this experiment to the lab queue in `config/production.json`
4. Run `grunt -prod`
5. Analyze the graphs and data (live-synced)


### Lab Results

<img alt="The dqn experiment analytics" src="http://kengz.me/openai_lab/images/dqn.png" />
<img alt="The dqn experiment analytics correlation" src="http://kengz.me/openai_lab/images/dqn_correlation.png" />

_The dqn experiment analytics generated by the Lab. This is a pairplot, where we isolate each variable, flatten the others, plot each trial as a point. The darker the color the higher ratio of the repeated sessions the trial solves._


|fitness_score|mean_rewards_per_epi_stats_mean|mean_rewards_stats_mean|epi_stats_mean|solved_ratio_of_sessions|num_of_sessions|max_total_rewards_stats_mean|t_stats_mean|trial_id|variable_gamma|variable_hidden_layers|variable_lr|
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
|5.305994917071314|1.3264987292678285|195.404|154.2|1.0|5|200.0|199.0|dqn-2017_03_19_004714_t79|0.999|[64]|0.02|
|5.105207228739003|1.2763018071847507|195.13600000000002|160.6|1.0|5|200.0|199.0|dqn-2017_03_19_004714_t50|0.99|[32]|0.01|
|4.9561426920909355|1.2390356730227339|195.26000000000002|168.6|1.0|5|200.0|199.0|dqn-2017_03_19_004714_t78|0.999|[64]|0.01|
|4.76714626254895|1.1917865656372375|195.106|172.4|1.0|5|200.0|199.0|dqn-2017_03_19_004714_t71|0.999|[32]|0.02|
|4.717243567762263|1.1793108919405657|195.56400000000002|167.2|1.0|5|200.0|199.0|dqn-2017_03_19_004714_t28|0.97|[32]|0.001|

_Analysis data table, top 5 trials._

On completion, from the analytics, we conclude that the experiment is a success, and the best agent that solves the problem has the parameters:

- *lr*: 0.02
- *gamma*: 0.999
- *hidden_layers_shape*: [64]


### Run Your Own Lab

Want to run the lab? Go to [Installation](http://kengz.me/openai_lab/installation), [Usage](http://kengz.me/openai_lab/usage) and [Development](http://kengz.me/openai_lab/development).
