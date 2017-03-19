# <a name="usage"></a>Usage

The general flow for running a production lab is:

1. Specify experiments in `rl/asset/experiment_specs.json`, e.g. `"dqn", "lunar_dqn"`
2. Specify the names of the experiments to run in `config/production.json`
3. Run the lab, e.g. `grunt -prod -resume`


## Commands

We use [Grunt](http://gruntjs.com/) to run the lab - set up experiments, pause/resume lab, run analyses, sync data, notify on completion. Internally `grunt` runs the `python` command, logged to stdout as `>> Composed command: python3 main.py ...`, which is harder to use.

The useful grunt commands are:

```shell
# when developing experiments specified in default.json
grunt

# run real lab experiments specified in production.json
grunt -prod
# run lab over ssh on remote server
grunt -prod -remote
# resume lab (previously incomplete experiments)
grunt -prod -remote -resume

# plot analysis graphs only
grunt analyze -prod

# clear data/ folder and cache files
grunt clear
```

See below for the full [Grunt Command Reference](#grunt-cmd) or the [Python Command Reference](#python-cmd).


**development** mode:

- All grunt commands defaults to this mode
- specify your dev experiment in `config/default.json`
- use only when developing your new algo
- the file-sync is in mock mode (emulated log without real file copying)
- no auto-notification


**production** mode:

- append the flag `-prod` to your `grunt` command
- specify your full experiments in `config/production.json`
- use when running experiments for real
- the file-sync is real
- has auto-notification to Slack channel


## Run Remotely

If you're using a remote server, run the commands inside a `screen`. That is, log in via ssh, start a screen, run, then detach screen.

```shell
screen -S lab
# enter the screen with the name "lab"
grunt -prod -remote -resume
# use Cmd+A+D to detach from screen, then Cmd+D to disconnect ssh
# to resume screen next time
screen -r lab
# use Cmd+D to terminate screen when lab ends
```

Since a remote server is away, you should check the system status occasionally to ensure no overrunning processes (memory leaks, stuck processes, overheating). Use [`glances`](https://github.com/nicolargo/glances) (already installed in `bin/setup`) to monitor your expensive machines.

<aside class="notice">
To monitor your system (CPU, RAM, GPU), run <code>glances</code>
</aside>

<img alt="Glances to monitor your system" src="./images/glances.png" />
_Glances on remote server beast._


## Resume Lab

Experiments take a long time to complete, and if your process gets terminated, resuming the lab is trivial with a `-resume` flag: `grunt -prod -remote -resume`. This will read the `config/history.json`:

```json
{
  "dqn": "dqn-2017_03_19_004714"
}
```

The `config/history.json` is created in the last run that maps `experiment_name`s to `experiment_id`s, and resume any incomplete experiments based on that `experiment_id`. You can manually tweak the file to set the resume target of course.



## <a name="grunt-cmd"></a>Grunt Command Reference

By default the `grunt` command (no task or flag) runs the lab in `development` mode using `config/default.json`.

The basic grunt command pattern is

```shell
grunt <task> -<flag>
```

The `<task>`s are:

- _(default empty)_: run the lab
- `analyze`: generate analysis data and graphs only, without running the lab. This can be used when you wish to see the analysis results midway during a long-running experiment. Run it on a separate terminal window as `grunt analyze -prod`
- `clear`: clear the `data/` folder and cache files. **Be careful** and make sure your data is already copied to the sync location


The `<flag>`s are:

- `-prod`: production mode, use `config/production.json`
- `-resume`: resume incomplete experiments from `config/history.json`
- `-remote`: when running over SSH, supplies this to use a fake display
- `-best`: run the finalized experiments with gym rendering and live plotting; without param selection. This uses the default `param` in `experiment_specs.json` that shall be updated to the best found.
- `-quiet`: mute all python logging in grunt. This is for lab-level development only.


## <a name="python-cmd"></a>Python Command Reference

The Python command is invoked inside `Gruntfile.js` under the `composeCommand` function. Change it if you need to.

The basic python command pattern is:

```shell
python3 main.py -<flag>

# most common example, with piping of terminal log
python3 main.py -bp -t 5 -e dqn | tee -a ./data/terminal.log;
```

The python command <flag>s are:

- `-a`: Run `analyze_experiment()` only to plot `experiment_data`. Default: `False`
- `-b`: blind mode, do not render graphics. Default: `False`
- `-d`: log debug info. Default: `False`
- `-e <experiment>`: specify which of `rl/asset/experiment_spec.json` to run. Default: `-e dev_dqn`. Can be a `experiment_name, experiment_id`.
- `-p`: run param selection. Default: `False`
- `-q`: quiet mode, log warning only. Default: `False`
- `-t <times>`: the number of sessions to run per trial. Default: `1`
- `-x <max_episodes>`: Manually specifiy max number of episodes per trial. Default: `-1` and program defaults to value in `rl/asset/problems.json`


## Lab Demo

Each experiment involves:
- a problem - an [OpenAI Gym environment](https://gym.openai.com/envs)
- a RL agent with modular components `agent, memory, optimizer, policy, preprocessor`, each of which is an experimental variable.

We specify input parameters for the experimental variable, run the experiment, record and analyze the data, conclude if the agent solves the problem with high rewards.

### Specify Experiment

The example below is fully specified in `rl/asset/classic_experiment_specs.json` under `dqn`:

```json
{
  "dqn": {
    "problem": "CartPole-v0",
    "Agent": "DQN",
    "HyperOptimizer": "GridSearch",
    "Memory": "LinearMemoryWithForgetting",
    "Optimizer": "AdamOptimizer",
    "Policy": "BoltzmannPolicy",
    "PreProcessor": "NoPreProcessor",
    "param": {
      "lr": 0.01,
      "decay": 0.0,
      "gamma": 0.99,
      "hidden_layers": [32],
      "hidden_layers_activation": "sigmoid",
      "exploration_anneal_episodes": 10
    },
    "param_range": {
      "lr": [0.001, 0.005, 0.01, 0.02],
      "gamma": [0.95, 0.97, 0.99, 0.999],
      "hidden_layers": [
        [16],
        [32],
        [64],
        [16, 8],
        [32, 16]
      ]
    }
  }
}
```

- *experiment*: `dqn`
- *problem*: [CartPole-v0](https://gym.openai.com/envs/CartPole-v0)
- *variable agent component*: `Boltzmann` policy
- *control agent variables*:
    - `DQN` agent
    - `LinearMemoryWithForgetting`
    - `AdamOptimizer`
    - `NoPreProcessor`
- *parameter variables values*: the `"param_range"` JSON

An **experiment** will run a trial for each combination of `param` values; each **trial** will run for multiple repeated **sessions**. For `dqn`, there are `4x4x5=80` param combinations (trials), and up to `5` repeated sessions per trial. Overall, this experiment will run at most `80 x 5 = 400` sessions.


### Lab Workflow

The workflow to setup this experiment is as follow:

1. Add the new theorized component `Boltzmann` in `rl/policy/boltzmann.py`
2. Specify `dqn` experiment spec in `experiment_spec.json` to include this new variable,  reuse the other existing RL components, and specify the param range.
3. Add this experiment to the lab queue in `config/production.json`
4. Run `grunt -prod`
5. Analyze the graphs and data (live-synced)


### Lab Results

<div style="max-width: 100%"><img alt="The dqn experiment analytics" src="./images/dqn.png" />
<br><br>
<img alt="The dqn experiment analytics correlation" src="./images/dqn_correlation.png" /></div>

_The dqn experiment analytics generated by the Lab. This is a pairplot, where we isolate each variable, flatten the others, plot each trial as a point. The darker the color the higher ratio of the repeated sessions the trial solves._


|fitness_score|mean_rewards_per_epi_stats_mean|mean_rewards_stats_mean|epi_stats_mean|solved_ratio_of_sessions|num_of_sessions|max_total_rewards_stats_mean|t_stats_mean|trial_id|variable_gamma|variable_hidden_layers|variable_lr|
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
|5.305994917071314|1.3264987292678285|195.404|154.2|1.0|5|200.0|199.0|dqn-2017_03_19_004714_t79|0.999|[64]|0.02|
|5.105207228739003|1.2763018071847507|195.13600000000002|160.6|1.0|5|200.0|199.0|dqn-2017_03_19_004714_t50|0.99|[32]|0.01|
|4.9561426920909355|1.2390356730227339|195.26000000000002|168.6|1.0|5|200.0|199.0|dqn-2017_03_19_004714_t78|0.999|[64]|0.01|
|4.76714626254895|1.1917865656372375|195.106|172.4|1.0|5|200.0|199.0|dqn-2017_03_19_004714_t71|0.999|[32]|0.02|
|4.717243567762263|1.1793108919405657|195.56400000000002|167.2|1.0|5|200.0|199.0|dqn-2017_03_19_004714_t28|0.97|[32]|0.001|

_Analysis data table, top 5 trials._

On completion, from the analytics, we conclude that the experiment is a success, and the best agent that solves the problem has the parameters:

- *lr*: 0.02
- *gamma*: 0.999
- *hidden_layers_shape*: [64]
