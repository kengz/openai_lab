# <a name="usage"></a>Usage

We use [Grunt](http://gruntjs.com/) to run the lab - set up experiments, pause/resume lab, run analyses, sync data, notify on completion. Internally `grunt` runs the `python` command which is harder to use, but we will include the details below for reference.

The general gist for running a production lab is:

1. Specify experiments in `rl/asset/experiment_specs.json`, e.g. `"dqn", "lunar_dqn"`
2. Specify the names of the experiments to run in `config/production.json`
3. Run the lab, e.g. `grunt -prod -resume`


## Commands

```shell
# when developing
grunt
# locally run lab experiments in production.json
grunt -prod
# run lab on remote server
grunt -prod -remote
# resume lab (previously incomplete experiments)
grunt -prod -remote -resume
# plot analysis graphs only
grunt plot -prod
```

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


<aside class="notice">
To monitor your system (CPU, RAM, GPU), run <code>glances</code> (which is installed in <code>bin/setup</code>)
</aside>

<img alt="Glances to monitor your system" src="./images/glances.png" />
_Glances on remote server beast._


### Run Remotely

If you're using a remote server, run the commands inside a screen. That is, log in via ssh, start a screen, run, then detach screen.

```shell
screen -S lab
# enter the screen with the name "lab"
grunt -prod -remote -resume
# use Cmd+A+D to detach from screen, then Cmd+D to disconnect ssh
# to resume screen next time
screen -r lab
# use Cmd+D to terminate screen when lab ends
```

### Resume Lab

Experiments take a long time to complete, and if your process gets terminated, resuming the lab is trivial with a `-resume` flag: `grunt -prod -remote -resume`

This will read the `config/history.json` created in the last run that maps `experiment_name`s to `experiment_id`s, and resume any incomplete experiments based on that `experiment_id`. You can manually tweak the file to set the resume target of course.

```json
{
  "dqn": "dqn-2017_02_21_182442"
}
```

### Internal Python command

The Python command is invoked inside `Gruntfile.js` under the `composeCommand` function. Change it if you need to.

Here's the pattern:

```shell
python3 main.py -bgp -e lunar_dqn -t 5 | tee -a ./data/terminal.log
```

The python command flags are:

- `-a`: Run `analyze_experiment()` only to plot `experiment_data`. Default: `False`
- `-b`: blind mode, do not render graphics. Default: `False`
- `-d`: log debug info. Default: `False`
- `-q`: quiet mode, log warning only. Default: `False`
- `-e <experiment>`: specify which of `rl/asset/experiment_spec.json` to run. Default: `-e dev_dqn`. Can be a `experiment_name, experiment_id`.
- `-g`: plot graphs live. Default: `False`
- `-m <max_evals>`: the max number of trials for hyperopt. Default: `100`
- `-p`: run param selection. Default: `False`
- `-t <times>`: the number of sessions to run per trial. Default: `1`
- `-x <max_episodes>`: Manually specifiy max number of episodes per trial. Default: `-1` and program defaults to value in `rl/asset/problems.json`
