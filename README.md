# OpenAI Lab [![CircleCI](https://circleci.com/gh/kengz/openai_lab.svg?style=shield)](https://circleci.com/gh/kengz/openai_lab) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/a0e6bbbb6c4845ccaab2db9aecfecbb0)](https://www.codacy.com/app/kengzwl/openai_lab?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=kengz/openai_lab&amp;utm_campaign=Badge_Grade)

[OpenAI Gym Doc](https://gym.openai.com/docs) | [OpenAI Gym Github](https://github.com/openai/gym) | [RL intro](https://gym.openai.com/docs/rl) | [RL Tutorial video Part 1](https://youtu.be/qBhLoeijgtA) | [Part 2](https://youtu.be/wNSlZJGdodE)

(Under work) An experimentation system for Reinforcement Learning using OpenAI and Keras.


## Installation

### Basic

```shell
git clone https://github.com/kengz/openai_lab.git
cd openai_lab
./bin/setup
```

Then, setup your `~/.keras/keras.json`. See example files in `config/keras.json`. We recommend Tensorflow for experimentation and multi-GPU, since it's much nicer to work with. Use Theano when you're training a single finalized model since it's faster.

The binary at `./bin/setup` installs all the needed dependencies, which includes the basic OpenAI gym, Tensorflow (for dev), Theano(for faster production), Keras.

*Note the Tensorflow is defaulted to CPU Mac or GPU Linux. [If you're on a different platform, choose the correct binary to install from TF.](https://www.tensorflow.org/get_started/os_setup#pip_installation)*

```shell
# default option
TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.12.1-py3-none-any.whl
# install from the TF_BINARY_URL
sudo pip3 install -U $TF_BINARY_URL
```


### Data files auto-sync (optional)

For auto-syncing `data/` files, we use Grunt. If you ran `./bin/setup` the dependencies for this are installed. This sets up a watcher for automatically copying data files via Dropbox. Set up a shared folder in your Dropbox and sync to desktop at the path `~/Dropbox/openai_lab/data`.

Also there's an automatic notification system that posts on your Slack when the experiment is completed. You'll need to install noti, get SLACK_TOKEN, set up config/default.json and Gruntfile.js. (More info soon when writing the API page).

```shell
# install if you haven't ran ./bin/setup
npm install --global grunt-cli
npm install
```


### Full OpenAI Gym Environments

To run more than just the classic control gym env, we need to install the OpenAI gym fully. We refer to the [Install Everything](https://github.com/openai/gym#installing-everything) of the repo (which is still broken at the time of writing).

```shell
brew install cmake boost boost-python sdl2 swig wget
git clone https://github.com/openai/gym.git
cd gym
pip3 install -e '.[all]'
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
python3 setup.py clean
python3 setup.py build
python3 setup.py install
```

To run Atari envs three additional dependencies are required

```shell
pip3 install atari_py
pip3 install Pillow
pip3 install PyOpenGL
```

Then check that it works with
```python
import gym
env = gym.make('SpaceInvaders-v0')
env.reset()
env.render()
```

## Usage

### (Pending unified command section) Run experiments

To set up the lab experiments, edit `config/default.json`.

```shell
# pure python command (you still need to know this to customize Grunt)
python3 main.py -bgp -e dev_dqn -t 2 | tee -a ./data/terminal.log
# run the lab in development mode (no watcher, no noti)
grunt
# run the lab in production mode
grunt -prod
# run the lab remotely
grunt -remote
# for development of an experiment, quick run
npm run dev
# run analysis only, even when shits half-running (in production, but don't need watcher)
grunt plot -e=dev_dqn_2017-02-12_183415
# if you're running it on a remote server
grunt plot -e=dev_dqn_2017-02-12_183415 -remote
```

### Running remotely

If you're running things remotely on a server, log in via ssh, start a screen, run, then detach screen.

```shell
screen -S run
# enter the screen with the name "run"
grunt remote
# use Cmd+A+D to detach from screen, then Cmd+D to disconnect ssh
# to resume screen next time
screen -r run
```

### Customize experiment commands

To customize your experiment commands, refer:

```shell
python3 main.py -bgp -e lunar_dqn -t 5 | tee -a ./data/terminal.log
```

The extra flags are:

- `-d`: log debug info. Default: `False`
- `-b`: blind mode, do not render graphics. Default: `False`
- `-e <experiment>`: specify which of `rl/asset/experiment_spec.json` to run. Default: `-e dev_dqn`. Can be a `experiment_name, experiment_id`.
- `-t <times>`: the number of sessions to run per trial. Default: `1`
- `-m <max_evals>`: the max number of trials: hyperopt max_evals to run. Default: `10`
- `-p`: run param selection. Default: `False`
- `-l`: run `line_search` instead of Cartesian product in param selection. Default: `False`
- `-g`: plot graphs live. Default: `False`
- `-a`: Run `analyze_experiment()` only to plot `experiment_data`. Default: `False`
- `-x <max_episodes>`: Manually specifiy max number of episodes per trial. Default: `-1` and program defaults to value in problems.json



## Development

This is still under active development, and documentation is sparse. The main code lives inside `rl/`.

The design of the code is clean enough to simply infer how things work by example.

- `data/`: data folders grouped per experiment, each of which contains all the graphs per trial sessions, JSON data file per trial, and csv metrics dataframe per run of multiple trials
- `rl/agent/`: custom agents. Refer to `base_agent.py` and `dqn.py` to build your own
- `rl/asset/`: specify new problems and experiment_specs to run experiments for.
- `rl/memory/`: RL agent memory classes
- `rl/policy/`: RL agent policy classes
- `rl/preprocessor/`: RL agent preprocessor (state and memory) classes
- `rl/analytics.py`: the data analytics module for output experiment data
- `rl/experiment.py`: the main high level experiment logic
- `rl/hyperoptimizer.py`: Hyperparameter optimizer for the Experiments
- `rl/util.py`: Generic util

Each run is an `experiment` that runs multiple `Trial`s (not restricted to the same `experiment_id` for future cross-training). Each `Trial` runs multiple (by flag `-t`) `Session`s, so an `trial` is a `sess_grid`.

Each trial collects the data from its sessions into `trial_data`, which is saved to a JSON and as many plots as there are sessions. On the higher level, `experiment` analyses the aggregate `trial_data` to produce a best-sorted CSV and graphs of the variables (what's changed across experiemnts) vs outputs.


## Roadmap

Check the latest under the [Github Projects](https://github.com/kengz/openai_lab/projects)

## Authors

- Wah Loon Keng
- Laura Graesser
