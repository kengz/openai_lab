# <a name="installation"></a>Installation

**1\.** **clone the repo**

`git clone https://github.com/kengz/openai_lab.git`

*If you plan to commit code, fork this repo then clone it instead.*


**2\.** **install dependencies**

Run the following commands to install:

- the system dependencies depending on your OS
- the project dependencies

*For quick repeated setup on remote servers, instead of these commands, run the equivalent setup script: `./bin/setup`*

```shell
# cd into project directory
cd openai_lab/

### MacOS System Dependencies
# Homebrew
ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
# OpenAI Gym dependencies
brew install cmake boost boost-python sdl2 swig wget
# noti
(curl -L https://github.com/variadico/noti/releases/download/v2.5.0/noti2.5.0.darwin-amd64.tar.gz | tar -xz); sudo mv noti /usr/local/bin/
# Node >= v7.0
brew install node
# Python >= v3.0
brew install python3

### Linux Ubuntu System Dependencies
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test && sudo apt-get update
sudo apt-get install -y gcc-4.9 g++-4.9 libhdf5-dev libopenblas-dev git python3-tk tk-dev
# OpenAI Gym dependencies
sudo apt-get install -y cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
# noti
(curl -L https://github.com/variadico/noti/releases/download/v2.5.0/noti2.5.0.linux-amd64.tar.gz | tar -xz); sudo mv noti /usr/local/bin/
# Node >= v7.0
(curl -sL https://deb.nodesource.com/setup_7.x | sudo -E bash -); sudo apt-get install -y nodejs
# Python >= v3.0
sudo apt-get -y install python3-dev python3-pip python3-setuptools


### Project Dependencies
./bin/copy-config
npm install; sudo npm i -g grunt-cli
# option 1: pip (ensure it is python3)
pip3 install -r requirements.txt
# option 2: virtualenv
virtualenv openai_lab
source openai_lab/bin/activate
pip3 install -r requirements.txt
# option 3: conda
conda env create -f environment.yml
source activate openai_lab
```


**3\.** **setup config files**

Run `./bin/copy-config`. This will create the config files from template, needed for lab [usage](#usage):

- `config/default.json` for local development, used when `grunt` is ran without a production flag.
- `config/production.json` for production lab run when `grunt -prod` is ran with the production flag `-prod`.


## <a name="quickstart"></a>Quickstart

The Lab comes with experiments with the best found solutions. Activate your installed environment and run your first below.


### Single Trial

Run the single best trial for an experiment using lab command: `grunt -best`

Alternatively, the plain python command invoked above is: `python3 main.py -e quickstart_dqn`

Then check your `./data/` folder for graphs and data files.

The grunt command is recommended as it's easier to schedule and run multiple experiments with. It sources from `config/default.json`, which should now have `quickstart_dqn`; more can be added.

```json
{
  "data_sync_destination": "~/Dropbox/openai_lab/data",
  "NOTI_SLACK_DEST": "#rl-monitor",
  "NOTI_SLACK_TOK": "GET_SLACK_BOT_TOKEN_FROM_https://my.slack.com/services/new/bot",
  "experiments": [
    "quickstart_dqn"
  ]
}
```

This trial is the best [found solution agent](https://github.com/kengz/openai_lab/pull/73) of `DQN` solving `Cartpole-v0`. You should see the Lab running like so:

![](./images/lab_demo_dqn.gif "Timelapse of OpenAI Lab")


### Experiment with Multiple Trials

Next step is to run a small experiment that searches for the best trial solutions.

```json
{
  "quickstart_dqn": {
    "problem": "CartPole-v0",
    "Agent": "DQN",
    ...
    "param_range": {
      "lr": [0.001, 0.01],
      "hidden_layers": [
        [32],
        [64]
      ]
    }
  }
}
```

This is under under `quickstart_dqn` in `rl/spec/classic_experiment_specs.json`. The experiment studies the effect of varying learning rate `lr` and the DQN neural net architecture `hidden_layers`. If you like, change the `param_range` to try more values.

Then, run: `grunt`

Alternatively the plain python command is: `python3 main.py -bp -e dqn`

Then check your `./data/` folder for graphs and data files.

The experiment will take about 15 minutes (depending on your machine). It will produce experiment data from the trials. Refer to [Analysis](#analysis) on how to interpret them.


### Next Up

We recommend:

- Continue reading below for the optional installation steps.
- [Solutions](#solutions) to see some existing solutions to start your agent from, as well as find environments/high scores to beat.
- [Agents](#agents) on how to create your agents from existing components, then add your own.
- [Usage](#usage) to continue reading the doc.


## Updating Lab

Check the Lab's latest [release versions here](https://github.com/kengz/openai_lab/releases).

- If you cloned directly the Lab, update with `git pull`
- If you forked, [setup a remote](https://help.github.com/articles/configuring-a-remote-for-a-fork/) and [update fork](https://help.github.com/articles/syncing-a-fork/)

```shell
# update direct clone
git pull

# update fork
git fetch upstream
git merge upstream/master
```


## Setup Data Auto-sync

We find it extremely useful to have data file-sync when running the lab on a remote server. This allows us to have a live view of the experiment graphs and data on our Dropbox app, on a computer or a smartphone.

For auto-syncing lab `data/` we use [Grunt](http://gruntjs.com/) file watcher for automatically copying data files to Dropbox. In your dropbox, set up a shared folder `~/Dropbox/openai_lab/data` and sync to desktop.

Setup the config key `data_sync_destination` in `config/{default.json, production.json}`.

<aside class="notice">
This step is optional; needed only when running production mode.
</aside>


## Setup Auto-notification

Experiments take a while to run, and we find it useful also to be notified automatically on completion. We use [noti](https://github.com/variadico/noti), which is also installed with `bin/setup`.

Set up a Slack, create a new channel `#rl_monitor`, and get a [Slack bot token](https://my.slack.com/services/new/bot).

Setup the config keys `NOTI_SLACK_DEST`, `NOTI_SLACK_TOK` in `config/{default.json, production.json}`.

<aside class="notice">
This step is optional; useful when running production mode.
</aside>

![](./images/noti.png "Notifications from the lab running on our remote server beast")
_Notifications from the lab running on our remote server beast._


## Hardware

For setting up your own hardware, especially with a GPU, googling will help more than we could. Also, setup is usually non-trivial since there're so many moving parts. Here's the recommended references:

- [A ~$1000 PC build](https://pcpartpicker.com/list/xdbWBP) (more expensive now ~$1200; buy your parts during Black Friday/sales.)
- [The official TensorFlow installation guide, with GPU setup info](https://www.tensorflow.org/install/install_linux)
- [Getting CUDA 8 to Work With openAI Gym on AWS and Compiling Tensorflow for CUDA 8 Compatibility](http://christopher5106.github.io/nvidia/2016/12/30/commands-nvidia-install-ubuntu-16-04.html)
- [Major OpenAI issue with SSH with xvfb failing with NVIDIA Driver due to opengl files](https://github.com/openai/gym/issues/366)
- [NVIDIA cannot install due to X server running](http://askubuntu.com/questions/149206/how-to-install-nvidia-run)
- [When login fails on Ubuntu after Nvidia installation](http://askubuntu.com/questions/759641/cant-get-nvidia-drivers-working-with-16-04-logs-out-right-after-login)
