# OpenAI Gym [![CircleCI](https://circleci.com/gh/kengz/openai_gym.svg?style=shield)](https://circleci.com/gh/kengz/openai_gym) [![Code Climate](https://codeclimate.com/github/kengz/openai_gym/badges/gpa.svg)](https://codeclimate.com/github/kengz/openai_gym)

[OpenAI Gym Doc](https://gym.openai.com/docs) | [OpenAI Gym Github](https://github.com/openai/gym) | [RL intro](https://gym.openai.com/docs/rl)

Working out at the (OpenAI) gym. **Note this is still under development, but will be ready before Nov 5**


## Installation

```shell
git clone https://github.com/kengz/openai_gym.git
cd openai_gym
python setup.py install
```

*Note that by default it installs Tensorflow for Python3 on MacOS. [Choose the correct binary to install from TF.](https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html#pip-installation)*

```shell
# for example, TF for Python2, MacOS
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.11.0rc1-py2-none-any.whl
sudo pip install --upgrade $TF_BINARY_URL
```

## Usage

Run the scripts inside the `rl/` folder. It will contain:
- a tour of the OpenAI gym
- a tabular q-learner
- a NN-based q-learner

To view **Tensorboard**, do `tensorboard --logdir='/tmp/tflearn_logs/'`


## Roadmap

- get the gym tour done
- clear the DQN class off the TF code, to make it backend-agnostic
- correct the basic q-learning algorithm (feels wrong)
- get the tabular q-learner working
- get NN q-learner working and solve the cartpole problem


## Authors

- Wah Loon Keng
- Laura Graesser
