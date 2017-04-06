# <a name="quickstart"></a>Quickstart

The Lab comes with experiments with the best found solutions. Run your first below.


### Single Trial

Run the single best trial for an experiment: `python3 main.py -e dqn`.

This is the best [found solution agent](https://github.com/kengz/openai_lab/pull/73) of `DQN` solving `Cartpole-v0`. You should see the rendering and graphs (check the `./data/` folder) like so:

![](./images/lab_demo_dqn.gif "Timelapse of OpenAI Lab")


### Experiment with Multiple Trials

Next step is to run a small experiment that searches for the best trial solutions.


```json
{
  "dqn": {
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


Go to `rl/spec/classic_experiment_specs.json` under `dqn` and modify the `param_range` to run less trials. This will study the effect of varying learning rate `lr` and the DQN neural net architecture `hidden_layers`.

Then, run `python3 main.py -bp -e dqn`. This will take about 15 minutes (depending on your machine).

It will produce experiment data from the trials. Refer to [Analysis](#analysis) on how to interpret them.


### Next Up

We recommend:

- [Solutions](#solutions) to see some existing solutions to start your agent from, as well as find environments/high scores to beat.
- [Agents](#agents) on how to create your agents from existing components, then add your own.
- [Usage](#usage) to continue reading the doc.
