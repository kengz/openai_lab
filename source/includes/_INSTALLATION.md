# <a name="installation"></a>Installation and Setup

```
git clone https://github.com/kengz/openai_lab.git
cd openai_lab
./bin/setup
```

`bin/setup` installs the dependencies the same way as our servers and CircleCI builds; inspect or change it as needed.

<aside class="notice">
By default `bin/setup` will install `tensorflow` for MacOS and `tensorflow-gpu` for Linux.
</aside>

Keras needs a backend in the home directory; setup your `~/.keras/keras.json` using example file in `config/keras.json`.

<aside class="notice">
We recommend **Tensorflow** for experimentation with multi-GPU for stability. Use **Theano** once your lab produces a final model for a single retraining, since it's faster.
</aside>

We use [Grunt](http://gruntjs.com/) to run the lab - set up experiments, pause/resume lab, run analyses, sync data, notify on progress. The related dependencies are installed with `bin/setup` already.

`bin/setup` also creates the needed config files:

- `config/default.json` for local development, used when `grunt` is ran without a production flag.
- `config/production.json` for production lab run when `grunt -prod` is ran with the production flag `-prod`


## Full OpenAI Gym Environments

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


## Data files auto-sync (optional)

We find it extremely useful to have data file sync when running the lab on a remote server. This allows us to have a live view of the experiment graphs and data on our Dropbox app on a computer or a smartphone.

For auto-syncing lab `data/` we use [Grunt](http://gruntjs.com/) file watcher for automatically copying data files via Dropbox. In your dropbox, set up a shared folder `~/Dropbox/openai_lab/data` and sync to desktop.

<aside class="notice">
Setup the config key `data_sync_destination` in `config/{default.json, production.json}`.
</aside>


## Notification

Experiments take a while to run, and we find it useful also to be notified automatically when it's complete. We use [noti](https://github.com/variadico/noti), which is also installed with `bin/setup`.

Set up a Slack, create a new channel `#rl_monitor`, and get a [Slack bot token](https://my.slack.com/services/new/bot).

<aside class="notice">
Setup the config keys `NOTI_SLACK_DEST`, `NOTI_SLACK_TOK` in `config/{default.json, production.json}`.
</aside>
