# <a name="usage"></a>Usage

*To understand the Lab's [Framework and Demo, skip to the next section.](#framework)*

The general flow for running a production lab is:

1. Specify experiment specs in `rl/spec/*_experiment_specs.json`, e.g. `"dqn", "lunar_dqn"`
2. Specify the names of the experiments to run in `config/production.json`
3. Run the lab, e.g. `grunt -prod -resume`


## Commands

We use [Grunt](http://gruntjs.com/) to run the lab - set up experiments, pause/resume lab, run analyses, sync data, notify on completion. Internally `grunt` runs the `python` command (harder to use), logged to stdout as `>> Composed command: python3 main.py ...`

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

- All grunt commands default to this mode
- specify your dev experiment in `config/default.json`
- use only when developing your new algorithms
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
# enter the screen with the name "lab"
screen -S lab
# run real lab over ssh, in resume mode
grunt -prod -remote -resume
# use Cmd+A+D to detach from screen, then Cmd+D to disconnect ssh

# to resume screen next time
screen -r lab
# use Cmd+D to terminate screen when lab ends
```

Since a remote server is away, you should check the system status occasionally to ensure no overrunning processes (memory growth, large processes, overheating). Use [`glances`](https://github.com/nicolargo/glances) (already installed in `bin/setup`) to monitor your expensive machines.

<aside class="notice">
To monitor your system (CPU, RAM, GPU), run <code>glances</code>
</aside>

<img alt="Glances to monitor your system" src="./images/glances.png" />
_Glances on remote server beast._


## Resume Lab

Experiments take a long time to complete, and if your process gets terminated, resuming the lab is trivial with a `-resume` flag: `grunt -prod -remote -resume`. This will use the `config/history.json`:

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
- `-e <experiment>`: specify which inside the `rl/spec/*_experiment_spec.json` to run. Default: `-e dev_dqn`. Can be a `experiment_name, experiment_id`.
- `-p`: run param selection. Default: `False`
- `-q`: quiet mode, log warning only. Default: `False`
- `-t <times>`: the number of sessions to run per trial. Default: `1`
- `-x <max_episodes>`: Manually specifiy max number of episodes per trial. Default: `-1` and program defaults to value in `rl/spec/problems.json`
