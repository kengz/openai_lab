---
title: OpenAI Lab Doc

language_tabs:
  - python

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
  - DEVELOPMENT
  - ROADMAP
  - CONTRIBUTING

search: true
---

# OpenAI Lab </br> [![CircleCI](https://circleci.com/gh/kengz/openai_lab.svg?style=shield)](https://circleci.com/gh/kengz/openai_lab) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/a0e6bbbb6c4845ccaab2db9aecfecbb0)](https://www.codacy.com/app/kengzwl/openai_lab?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=kengz/openai_lab&amp;utm_campaign=Badge_Grade) [![Codacy Badge](https://api.codacy.com/project/badge/Coverage/9e55f845b10b4b51b213620bfb98e4b3)](https://www.codacy.com/app/kengzwl/openai_lab?utm_source=github.com&utm_medium=referral&utm_content=kengz/openai_lab&utm_campaign=Badge_Coverage) [![GitHub forks](https://img.shields.io/github/forks/kengz/openai_lab.svg?style=social&label=Fork)](https://github.com/kengz/openai_lab) [![GitHub stars](https://img.shields.io/github/stars/kengz/openai_lab.svg?style=social&label=Star)](https://github.com/kengz/openai_lab)

**(DOC UNDER CONSTRUCTION)**

_An experimentation system for Reinforcement Learning using OpenAI and Keras._

The _OpenAI Lab_ is created to do Reinforcement Learning (RL) like science - _theorize, experiment_. It provides an easy to use interface to [OpenAI Gym](https://gym.openai.com/) and [Keras](https://keras.io/), combined with an automated experimental and analytics framework.

While these are powerful tools, they take a lot to get running. Of many implementations we saw which solve OpenAI gym environments, many had to rewrite the same basic components instead of just the new components being researched.

To address this, the Lab does three things.

1. Handles the basic RL environment and algorithm setups.
2. Provides a standard, extensible platform with reusable components for developing deep reinforcement learning algorithms.
3. Provides a rigorous experimentation system with logs, plots and analytics for testing new RL algorithms. Experimental settings are logged in a standardized format so that solutions can be reproduced by anyone using the Lab.

With OpenAI Lab, we could focus on researching the essential elements of reinforcement learning such as the algorithm, policy, memory (experience replay), and parameter tuning to solve the OpenAI environments. We could also test our hypotheses more reliably.

See the [Best Solutions](#solutions) to some OpenAI environments that OpenAI Lab users have produced.


### Run the Lab

Next, see [Installation](#installation) and [Usage](#usage).


<div style="max-width: 100%"><img alt="Timelapse of OpenAI Lab" src="./images/lab_demo_dqn.gif" /></div>

*Timelapse of OpenAI Lab, solving CartPole-v0.*

