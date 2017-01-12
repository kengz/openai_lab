# OpenAI Gym [![CircleCI](https://circleci.com/gh/kengz/openai_gym.svg?style=shield)](https://circleci.com/gh/kengz/openai_gym) [![Code Climate](https://codeclimate.com/github/kengz/openai_gym/badges/gpa.svg)](https://codeclimate.com/github/kengz/openai_gym)

[OpenAI Gym Doc](https://gym.openai.com/docs) | [OpenAI Gym Github](https://github.com/openai/gym) | [RL intro](https://gym.openai.com/docs/rl)

Working out at the (OpenAI) gym.

This was started from the Eligible Reinforcement Learning event, and we just kept working on it. Checkout the tutorial session videos here: [Part 1](https://youtu.be/qBhLoeijgtA), [Part 2](https://youtu.be/wNSlZJGdodE)


## Installation

### Basic

```shell
git clone https://github.com/kengz/openai_gym.git
cd openai_gym
python setup.py install
```

*Note that by default it installs Tensorflow for Python3 on MacOS. [Choose the correct binary to install from TF.](https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html#pip-installation)*

```shell
# for example, TF for Python2, MacOS
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.11.0rc1-py2-none-any.whl
# or Linux CPU-only, Python3.5
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0rc1-cp35-cp35m-linux_x86_64.whl
sudo pip install --upgrade $TF_BINARY_URL
```

### Complete

To run more than just the classic control gym env, we need to install the OpenAI gym fully. We refer to the [Install Everything](https://github.com/openai/gym#installing-everything) of the repo (which is still broken at the time of writing).

```shell
brew install cmake boost boost-python sdl2 swig wget
git clone https://github.com/openai/gym.git
cd gym
pip install -e '.[all]'
```

Try to run a Lunar Lander env, it will break (unless they fix it):
```python
import gym
env = gym.make('LunarLander-v2')
env.reset()
env.render()
```

If it fails, debug as follow (and repeat once more if it fails again, glorious python):

```shell
pip3 uninstall Box2D box2d-py
git clone https://github.com/pybox2d/pybox2d
cd pybox2d/
python setup.py clean
python setup.py build
python setup.py install
```

To run Atari envs three additional dependencies are required

```shell
pip install atari_py
pip install Pillow
pip install PyOpenGL
```

Then check that it works with
```python
import gym
env = gym.make('SpaceInvaders-v0')
env.reset()
env.render()
```

## Usage

The scripts are inside the `rl/` folder. Configure your `game_specs` in `rl/session.py`, and run

```shell
python main.py # run in normal mode
python main.py -d # print debug log
python main.py -b # blind, i.e. don't render graphics
python main.py 2>&1 | tee run.log # write to log file
```

**Ubuntu Debug**: If shits still blow up with `pyglet.canvas.xlib.NoSuchDisplayException: Cannot connect to "None"` even with the `-b` no-render flag, prepend a `xvfb-run` command like so:

```shell
sudo apt-get install -y xvfb
xvfb-run -a -s "-screen 0 1400x900x24" -- python main.py -d -b
```

Each run is an experiment, and data will be collected and written once every experiment is finished to `<date>_data_grid.json`. If rendering is enabled, it will also save the graph to `<Problem>.png`.


### Hyperparam Selection

1. Define the `game_specs.<game>.param_range` in `rl/session.py` for hyperparameter selection, which will run using multiprocessing, and return the best param (defined as having the best average session `mean_rewards`).

2. Modify `main.py` to run single experiment or param selection multi experiments, each for a number of times `run(<game>, run_param_selection=False, times=1)`.

3. Run `python main.py` as before. At the end it will return and print out a ranked (from the best) list of params.

Note that in parallel mode, graphics will not be rendered.


## Development

See `rl/session.py` for the main Session class. It takes the 3 arguments `Agent, problem, param`.

You should implement your `Agent` class in the `rl/agent/` folder. By polymorphism, your `Agent` should implement the methods shown in `rl/agent/base_agent/py`, otherwise it will throw `NotImplementedError`.

```python
# see rl/agent/dqn.py for example
def __init__(self, env_spec, *args, **kwargs):
    super(DQN, self).__init__(env_spec)
    # set other params for agent
def select_action(self, state):
    self.policy.select_action(state)
def update(self, sys_vars):
    self.policy.update(sys_vars)
def to_train(self, sys_vars):
    return True  # if train at every step
def train(self, sys_vars):
    # training code...
```

Refer to the following Agents in `rl/agent/` for building your own:
- `dummy.py`: dummy agent used for a gym tour. Does random actions.
- `q_table.py`: a tabular q-learner
- `dqn.py`: agent with a simple Deep Q-Network, forms the base `DQN` class for others to inherit
- `double_dqn.py`: agent with Double Deep Q-Network, inherits `DQN`
- `lunar_dqn.py`: agent with deeper network for the Lunar-Lander game, inherits `DQN`
- `lunar_double_dqn.py`: agent with deeper network for the Lunar-Lander game, inherits `DoubleDQN`


## Roadmap

Check the latest under the [Github Projects](https://github.com/kengz/openai_gym/projects)

## Authors

- Wah Loon Keng
- Laura Graesser
