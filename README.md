# OpenAI Lab [![CircleCI](https://circleci.com/gh/kengz/openai_lab.svg?style=shield)](https://circleci.com/gh/kengz/openai_lab) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/9e55f845b10b4b51b213620bfb98e4b3)](https://www.codacy.com/app/kengzwl/openai_lab?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=kengz/openai_lab&amp;utm_campaign=Badge_Grade) [![Codacy Badge](https://api.codacy.com/project/badge/Coverage/9e55f845b10b4b51b213620bfb98e4b3)](https://www.codacy.com/app/kengzwl/openai_lab?utm_source=github.com&utm_medium=referral&utm_content=kengz/openai_lab&utm_campaign=Badge_Coverage) [![GitHub forks](https://img.shields.io/github/forks/kengz/openai_lab.svg?style=social&label=Fork)](https://github.com/kengz/openai_lab) [![GitHub stars](https://img.shields.io/github/stars/kengz/openai_lab.svg?style=social&label=Star)](https://github.com/kengz/openai_lab)

---

<p align="center"><b><a href="http://kengz.me/openai_lab">OpenAI Lab Documentation</a></b></p>

---

_An experimentation system for Reinforcement Learning using OpenAI Gym, Tensorflow, and Keras._

The _OpenAI Lab_ is created to do Reinforcement Learning (RL) like science - _theorize, experiment_. It provides an easy to use interface to [OpenAI Gym](https://gym.openai.com/) and [Keras](https://keras.io/), combined with an automated experiment and evaluation framework.

This is [motivated by the problems we experienced in RL research](http://kengz.me/openai_lab/#motivations): the difficulty of building upon other's work, the lack of rigor in comparisons of research results, and the inertia to high level vision.

The Lab aims to make RL research more efficient and to encourage experimentation, by doing three things:

1. Handles the basic RL environment and algorithm setups.
2. Provides a standard, extensible platform with reusable components for developing deep reinforcement learning algorithms.
3. Provides a rigorous experimentation system with logs, plots and analytics for testing new RL algorithms. Experimental settings are logged in a standardized format so that solutions can be reproduced by anyone using the Lab.

With OpenAI Lab, we could focus on researching the essential elements of reinforcement learning such as the algorithm, policy, memory, and parameter tuning. It allows us to build agents efficiently using existing components with the implementations from research ideas. We could then test the research hypotheses systematically by running experiments.

See the [Solutions](http://kengz.me/openai_lab/#solutions) to some OpenAI environments that the Lab users have produced.

*Ultimately, the Lab is a generalized framework for doing reinforcement learning, agnostic of OpenAI Gym and Keras. Pytorch-based implementations are on the roadmap, for example.*


<img alt="Timelapse of OpenAI Lab" src="http://kengz.me/openai_lab/images/lab_demo_dqn.gif" />
_Timelapse of OpenAI Lab, solving CartPole-v0._
