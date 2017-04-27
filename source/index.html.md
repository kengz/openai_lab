---
title: OpenAI Lab Doc

language_tabs:

toc_footers:
  - <a href='https://github.com/kengz/openai_lab'>OpenAI Lab Github</a>
  - <a href='https://github.com/openai/gym'>OpenAI Gym Github</a>
  - <a href='https://github.com/fchollet/keras'>Keras Github</a>
  - <a href='https://youtu.be/qBhLoeijgtA'>RL Tutorial video part 1/2</a>
  - <a href='https://youtu.be/wNSlZJGdodE'>RL Tutorial video part 2/2</a>

includes:
  - INSTALLATION
  - USAGE
  - EXPERIMENTS
  - ANALYSIS
  - SOLUTIONS
  - METRICS
  - ALGORITHMS
  - AGENTS
  - DEVELOPMENT
  - CONTRIBUTING
  - MOTIVATIONS

search: true
---

# OpenAI Lab </br> [![GitHub release](https://img.shields.io/github/release/kengz/openai_lab.svg)](https://github.com/kengz/openai_lab) [![CircleCI](https://circleci.com/gh/kengz/openai_lab.svg?style=shield)](https://circleci.com/gh/kengz/openai_lab) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/9e55f845b10b4b51b213620bfb98e4b3)](https://www.codacy.com/app/kengzwl/openai_lab?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=kengz/openai_lab&amp;utm_campaign=Badge_Grade) [![Codacy Badge](https://api.codacy.com/project/badge/Coverage/9e55f845b10b4b51b213620bfb98e4b3)](https://www.codacy.com/app/kengzwl/openai_lab?utm_source=github.com&utm_medium=referral&utm_content=kengz/openai_lab&utm_campaign=Badge_Coverage) [![GitHub stars](https://img.shields.io/github/stars/kengz/openai_lab.svg?style=social&label=Star)](https://github.com/kengz/openai_lab) [![GitHub forks](https://img.shields.io/github/forks/kengz/openai_lab.svg?style=social&label=Fork)](https://github.com/kengz/openai_lab)

_An experimentation system for Reinforcement Learning using OpenAI Gym, Tensorflow, and Keras._

_OpenAI Lab_ is created to do Reinforcement Learning (RL) like science - _theorize, experiment_. It provides an easy interface to [OpenAI Gym](https://gym.openai.com/) and [Keras](https://keras.io/), with an automated experiment and evaluation framework.

This is [motivated by the problems we faced in RL research](#motivations): *the difficulty of building upon other's work, the lack of rigor in comparisons of research results, and the inertia to high level vision.*

The Lab aims to make RL research more efficient and to encourage experimentation, by doing three things:

1. Handles the basic RL environment and algorithm setups.
2. Provides a standard, extensible platform with reusable components for developing deep reinforcement learning algorithms.
3. Provides a rigorous experimentation system with logs, plots and analytics for testing new RL algorithms. Experimental settings are stored in standardized format for reproducibility and comparisons.

With OpenAI Lab, we could focus on researching the essential elements of reinforcement learning such as the algorithm, policy, memory, and parameter tuning. It allows us to build agents efficiently using existing components with the implementations from research ideas. We could then test the research hypotheses systematically by running experiments.

*Ultimately, the Lab is a generalized framework for doing reinforcement learning, agnostic of OpenAI Gym and Keras. Pytorch-based implementations are on the roadmap, for example.*

### Implemented Algorithms

See the **[Fitness Matrix](#fitness-matrix)** for the results of these algorithms below in the OpenAI gym environments.

|algorithm|implementation|eval score (pending)|
|:---|:---|:---|
|[DQN](https://arxiv.org/abs/1312.5602)|[DQN](https://github.com/kengz/openai_lab/blob/master/rl/agent/dqn.py)|-|
|[Double DQN](https://arxiv.org/abs/1509.06461)|[DoubleDQN](https://github.com/kengz/openai_lab/blob/master/rl/agent/double_dqn.py)|-|
|[Dueling DQN](https://arxiv.org/abs/1511.06581)|-|-|
|Sarsa|[DeepSarsa](https://github.com/kengz/openai_lab/blob/master/rl/agent/deep_sarsa.py)|-|
|Off-Policy Sarsa|[OffPolicySarsa](https://github.com/kengz/openai_lab/blob/master/rl/agent/offpol_sarsa.py)|-|
|[PER (Prioritized Experience Replay)](https://arxiv.org/abs/1511.05952)|[PrioritizedExperienceReplay](https://github.com/kengz/openai_lab/blob/master/rl/memory/prioritized_exp_replay.py)|-|
|[CEM (Cross Entropy Method)](https://en.wikipedia.org/wiki/Cross-entropy_method)|next|-|
|[REINFORCE](http://incompleteideas.net/sutton/williams-92.pdf)|-|-|
|[DPG (Deterministic Policy Gradient) off-policy actor-critic](http://jmlr.org/proceedings/papers/v32/silver14.pdf)|[ActorCritic](https://github.com/kengz/openai_lab/blob/master/rl/agent/actor_critic.py)|-|
|[DDPG (Deep-DPG) actor-critic with target networks](https://arxiv.org/abs/1509.02971)|[DDPG](https://github.com/kengz/openai_lab/blob/master/rl/agent/ddpg.py)|-|
|[A3C (asynchronous advantage actor-critic)](https://arxiv.org/pdf/1602.01783.pdf)|-|-|
|Dyna|next|-|
|[TRPO](https://arxiv.org/abs/1502.05477)|-|-|
|Q*(lambda)|-|-|
|Retrace(lambda)|-|-|
|[Neural Episodic Control (NEC)](https://arxiv.org/abs/1703.01988)|-|-|
|[EWC (Elastic Weight Consolidation)](https://arxiv.org/abs/1612.00796)|-|-|


### Run the Lab

Next, see [Installation](#installation) and jump to [Quickstart](#quickstart).


<div style="max-width: 100%"><img alt="Timelapse of OpenAI Lab" src="./images/lab_demo_dqn.gif" /></div>

*Timelapse of OpenAI Lab, solving CartPole-v0.*

