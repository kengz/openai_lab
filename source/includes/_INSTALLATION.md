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
# Node >= v7.0 and dependencies
brew install node
# Python >= v3.0 and dependencies
brew install python3

### Linux Ubuntu System Dependencies
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test && sudo apt-get update
sudo apt-get install -y gcc-4.9 g++-4.9 libhdf5-dev libopenblas-dev git
# OpenAI Gym dependencies
sudo apt-get install -y cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
# noti
(curl -L https://github.com/variadico/noti/releases/download/v2.5.0/noti2.5.0.linux-amd64.tar.gz | tar -xz); sudo mv noti /usr/local/bin/
# Node >= v7.0 and dependencies
(curl -sL https://deb.nodesource.com/setup_7.x | sudo -E bash -); sudo apt-get install -y nodejs
# Python >= v3.0 and dependencies
sudo apt-get -y install python3-dev python3-pip python3-setuptools

### Project Dependencies
npm install; sudo npm i -g grunt-cli
# with pip/virtualenv
sudo pip3 install -r requirements.txt
# with conda
while read requirement; do conda install --yes $requirement; done < requirements.txt
```


**3\.** **setup config files**

Create the config files from template: `./bin/copy-config`


**2\.** Keras needs a **backend file** in your home directory; setup `~/.keras/keras.json` using the example file in `config/keras.json`.

```json
{
  "epsilon": 1e-07,
  "image_dim_ordering": "tf",
  "floatx": "float32",
  "backend": "tensorflow"
}
```


**3\.** `bin/setup` also creates the needed **config files** needed for lab [usage](#usage). See sections below for more info.

- `config/default.json` for local development, used when `grunt` is ran without a production flag.
- `config/production.json` for production lab run when `grunt -prod` is ran with the production flag `-prod`.

```json
{
  "data_sync_destination": "~/Dropbox/openai_lab/data",
  "NOTI_SLACK_DEST": "#rl-monitor",
  "NOTI_SLACK_TOK": "GET_SLACK_BOT_TOKEN_FROM_https://my.slack.com/services/new/bot",
  "experiments": [
    "dev_dqn",
    "dqn"
  ]
}
```


### Updating Lab

Check the [release versions here](https://github.com/kengz/openai_lab/releases).

- If you cloned directly the Lab, update with `git pull`
- If you forked, [setup a remote](https://help.github.com/articles/configuring-a-remote-for-a-fork/) and [update fork](https://help.github.com/articles/syncing-a-fork/).


### Jump to Quickstart

Come back to the optional installation steps below when you need them later.

Next, jump to [Quickstart](#quickstart) to run your first experiment.


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


## Setup Experiments

There are many existing experiments specified in `rl/spec/*_experiment_specs.json`, and you can add more. Pick the `experiment_name`s (e.g. `"dqn", "lunar_dqn"`), specify in `config/default.json` or `config/production.json`. Then check [usage](#usage) to run the lab.


## Hardware

For setting up your own hardware, especially with a GPU, googling will help more than we could. Also, setup is usually non-trivial since there're so many moving parts. Here's the recommended references:

- [A ~$1000 PC build](https://pcpartpicker.com/list/xdbWBP) (more expensive now ~$1200; buy your parts during Black Friday/sales.)
- [The official TensorFlow installation guide, with GPU setup info](https://www.tensorflow.org/install/install_linux)
- [Getting CUDA 8 to Work With openAI Gym on AWS and Compiling Tensorflow for CUDA 8 Compatibility](http://christopher5106.github.io/nvidia/2016/12/30/commands-nvidia-install-ubuntu-16-04.html)
- [Major OpenAI issue with SSH with xvfb failing with NVIDIA Driver due to opengl files](https://github.com/openai/gym/issues/366)
- [NVIDIA cannot install due to X server running](http://askubuntu.com/questions/149206/how-to-install-nvidia-run)
- [When login fails on Ubuntu after Nvidia installation](http://askubuntu.com/questions/759641/cant-get-nvidia-drivers-working-with-16-04-logs-out-right-after-login)
