# <a name="installation"></a>Installation

1\. Clone repo and run the setup script:

```shell
git clone https://github.com/kengz/openai_lab.git
cd openai_lab
./bin/setup
```

`bin/setup` installs all the dependencies the same way as our servers and [CircleCI builds](https://circleci.com/gh/kengz/openai_lab); inspect or change it as needed. Also make sure your dependencies are the most updated - check the [major required versions here](#dependencies)

<aside class="notice">
If you use OpenAI Lab for serious experimentations, forking this repo then clone your fork, so you can commit code and even contribute to the Lab.
</aside>

2\. Keras needs a backend in the home directory; setup your `~/.keras/keras.json` using example file in `config/keras.json`.

```json
{
  "epsilon": 1e-07,
  "image_dim_ordering": "tf",
  "floatx": "float32",
  "backend": "tensorflow"
}
```

<aside class="notice">
We recommend Tensorflow for experimentation with multi-GPU for stability. By default <code>bin/setup</code> will install <code>tensorflow</code> for MacOS and <code>tensorflow-gpu</code> for Linux.
Use Theano once your lab produces a final model for a single retraining, since it's faster.
</aside>


3\. `bin/setup` also creates the needed config files needed for lab [usage](#usage). See sections below for more info.

- `config/default.json` for local development, used when `grunt` is ran without a production flag.
- `config/production.json` for production lab run when `grunt -prod` is ran with the production flag `-prod`

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


## Setup Data Auto-sync

We find it extremely useful to have data file sync when running the lab on a remote server. This allows us to have a live view of the experiment graphs and data on our Dropbox app on a computer or a smartphone.

For auto-syncing lab `data/` we use [Grunt](http://gruntjs.com/) file watcher for automatically copying data files via Dropbox. In your dropbox, set up a shared folder `~/Dropbox/openai_lab/data` and sync to desktop.

<aside class="notice">
Setup the config key <code>data_sync_destination</code> in <code>config/{default.json, production.json}</code>.
</aside>


## Setup Auto-notification

Experiments take a while to run, and we find it useful also to be notified automatically when it's complete. We use [noti](https://github.com/variadico/noti), which is also installed with `bin/setup`.

Set up a Slack, create a new channel `#rl_monitor`, and get a [Slack bot token](https://my.slack.com/services/new/bot).

<aside class="notice">
Setup the config keys <code>NOTI_SLACK_DEST</code>, <code>NOTI_SLACK_TOK</code> in <code>config/{default.json, production.json}</code>.
</aside>

<img alt="Notifications from the lab running on our remote server beast" src="./images/noti.png" />
_Notifications from the lab running on our remote server beast._


## Setup Experiments

There are many existing experiments specified in `rl/asset/experiment_specs.json`, and you can add more. Pick the `experiment_name`s (the JSON key, e.g. `"dqn", "lunar_dqn"`), list under config key `experiment`. Then check [usage](#usage) to run the lab.


## <a name="dependencies"></a>List of Dependencies

There is more than a dozen of dependencies. For the full list, inspect `bin/setup`. Here are some major ones and their minimum required versions. If lab fails to run, check these first:

- `python3 >= 3.4`
- `node >= 7.0`
- `tensorflow >= 1.0` or `tensorflow-gpu >= 1.0`
- `theano == 0.8.2`
- `keras >= 1.2`
- `gym[all] >= 0.7`
