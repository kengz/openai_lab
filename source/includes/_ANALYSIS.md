# <a name="analysis"></a>Analysis

Once the Lab is running experiments, it will produce data. This section details how to analyze and understand the data, before we can contribute to the [Best Solutions](#solutions).

An experiment produces 3 types of data files in the folder `data/<experiment_id>/`:

- **session plots**: `<session_id>.png`
- **trial_data**: `<trial_id>.json`
- **experiment data**:
    - `<experiment_id>_analysis_data.csv`
    - `<experiment_id>_analysis.png`
    - `<experiment_id>_analysis_correlation.png`

<aside class="notice">
Refer to <a href="#structure">Experiments > Structure</a> to see how the files are produced.
</aside>


We will illustrate with an example experiment from the [Lab Demo](#demo), with the actual data from the [dqn solution PR](https://github.com/kengz/openai_lab/pull/73).


## Session Graphs

><img alt="The best session graph" src="https://cloud.githubusercontent.com/assets/8209263/24180935/404370ea-0e8e-11e7-8f20-f8691ee03e7b.png" />
>*The best session graph from the [dqn-2017_03_19_004714](https://github.com/kengz/openai_lab/pull/73) experiment. From the session graph we can see that the agent starts learning the CartPole-v0 task at around episode 15, then solves it before episode 20. Over time the loss decreases, the solution becomes stabler, and the mean rewards increases until the session is solved reliably.*


When an experiment is running, the lab will plot the session graphs live, one for each session.

### How to read

A session graph has 3 subplots:

1. **total rewards and exploration rate vs episode**: directly records the (blue) total rewards attained at the end of each episode, in relation to the (red) exploration rate (`epsilon`, `tau`, etc. depending on the policy).

    The 2 lines usually show negative correlation - when the exploration rate drops, the total rewards should rise. When a solution becomes stable, the blue line should stay around its max.


2. **mean rewards over the last 100 episodes vs episode**: measures the 100-episode mean of the total rewards from above.

    Defined by OpenAI, this metric is usually how a solution is identified - when it hits a target solution score, which would mean that the solution is sufficiently *strong* and *stable*.


3. **loss vs time**: measures the loss of the agent's neural net. This graph is all the losses  concatenated over time, over all episodes.

    There is no specific unit for the loss as it depends on what loss function is used in the NN architecture (typically `mean_squared_error`). As the NN starts getting more accurate, the loss should decrease.


<aside class="notice">
When developing a new algorithm, use the session graph to immediately see how the agent is performing without needing to wait for the entire session to complete.
</aside>


## Analysis Graph

><img alt="The dqn experiment analytics" src="https://cloud.githubusercontent.com/assets/8209263/24087747/41a86170-0cf9-11e7-84b8-8f3fcae24c95.png" />
>*The analysis graph from the [dqn-2017_03_19_004714](https://github.com/kengz/openai_lab/pull/73) experiment. There're numerous dark points with solved_ratio 1, which is expected since CartPole-v0 is the simplest environment. There are clear trends cross the x-values - gamma=0.95 is unstable; 2-hidden-layer NN is unsuitable for the problem, but wider 1-hidden-layer is good; learning rate lr=0.001 is stabler, but lr=0.02 is a good balance between stability and fitness_score.*


The **analysis graph** is the primary graph used to judge the overall experiment - how all the trials perform. It is a pair-plot of the *measurement metrics on the y-axis*, and the *experiment variables on the x-axis*.

*(new adjacent possible)*

### The y-axis measurement metrics

1. `fitness_score`: the final evaluation metric the Lab uses to select a fit agent (an agent with the fit parameter set for that class of Agent). The design and purpose of it is more involved - see [metrics](#metrics) for more.

2. `mean_rewards_stats_mean`: the statistical mean of all the `mean_rewards` over all the sessions of a trial. Measures the average solution potential of a trial.

3. `max_total_rewards_stats_mean`: the statistical mean of all the `max_total_rewards` over all the sessions of a trial. Measures the agent's average peak performance.

4. `epi_stats_mean`: the statistical mean of the termination episode of a session. The lower the better, as it would imply that the agent solves the environment faster on average.


### The hue metrics

Each data point represents a trial, with the data averaged over its sessions. The points are colored (see legend) with the hue:

- `solved_ratio_of_sessions`: how many sessions are solved out of the total sessions in a trial. 0 means none, 1 means all.

The granularity of the `solved_ratio_of_sessions` depends on the number of sessions ran per trial. From experience, we settle on 5 sessions per trial as it's the best tradeoff between granularity and computation time.


*(new adjacent possible)*

Multiple sessions allow us to observe the consistency of an agent. As we have noticed across the parameter space, there is a spectrum of solvability: agents who cannot solve at all, can solve occasionally, and can always solve. The agents that solves occasionally can be valuable when developing an new algorithm, and most people will throw them away - this is bad when a strong agent is hard to find in the early stage.


### How to read

Every subplot in the graph shows the distribution of all the trial points in the pair of *y vs x* variables, with the other *x'* dimensions flattened. For each, observe the population distribution, y-positions, and trend across the x-axis.

Note that these will use [swarmplot](http://seaborn.pydata.org/generated/seaborn.swarmplot.html) which allows us to see the distribution of points by spreading them horizontally to prevent overlap. However, when the x-axis has too many values (.e.g continuous x-values in random search), it will switch to scatter plot instead.

**Population distribution**: more darker points implies that the many trials could solve the environment consistently. Higher ratio of dark points also means the environment is easier for the agent. If the points are closer and the distribution has smaller vertical gaps, then the *x* is a stabler value for the *y* value even when other *x'* dimensions vary. In a scatterplot, clustering of points in a random search also shows the convergence of the search.

**trend across y-values**: the fitter trial will show up higher in the y-axes (except for `epi_stats_mean`). Generally good solutions are scarce and they show up at higher `fitness_score`, whereas the non-solutions get clustered in the lower region. Notice how the `fitness_score` plots can clearly distinguish the good solutions (darker points), whereas in the `mean)rewards_stats_mean` and `max_total_rewards_stats_mean` plots it is hard to tell apart. We will discuss how the custom-formulated `fitness_score` function achieves this in the [metrics](#metrics) section.

**trend across x-values**: to find a stable and good *x-value*, observe the vertical gaps in distribution, the clustering of darker points. Usually there's one maxima with a steady trend towards it. Recall that the plots flatten the other *x'* values, but the dependence on *x* value is usually very consistent across *x'* that there will still be a flattened trend.


## Correlation Graph

><img alt="The dqn experiment analytics correlation" src="https://cloud.githubusercontent.com/assets/8209263/24087746/418a9b54-0cf9-11e7-8aac-f0df817def43.png" />

>*The correlation graph from the [dqn-2017_03_19_004714](https://github.com/kengz/openai_lab/pull/73) experiment. We can see smooth contours of spectrum in them, suggesting that the x-values are stable - small change in values will not be catastrophic. There are 2 darker regions in the contour; the distribution confirms that gamma=0.999 and lower lr are indeed stabler, since they have higher populations of darker points. The instability of gamma=0.95 shows clearly as 2 segments of stacked bar with huge contrast.*

*(new adjacent possible)*

The **correlation graph** reveals pairwise x-value correlations that is flattened in the analysis graph. This is a pair-plot between the orderable parameter variables.


### How to read

The diagonals simply shows the population distribution for that x-value; the diagonal plots show the contoured spectrum that tells how to best combine separate parameter values.

The points are semi-transparent, so if they overlap, their colors will stack instead of hiding the points behind.



## Data

After glancing through the graphs, it will be easier to understand the data and find the targets.

### How to read

The `<experiment_id>_analysis_data.csv` will show the data for each trial, sorted by the highest `fitness_score` first. The left columns are the measured output values; then they're separated by the `trial_id`; the right columns are the parameter values for the trial.

The `trial_id` will tell us which `trial_data` to check for even more details on the best trials. Usually we can also spot some trend in the right parameter columns.

The best `<trial_id>.json` will show us directly what is its `experiment_spec`, and more stats about the trial. When [submitting a solution PR](#solutions), retrieve the `experiment_spec` to update the default `*_experiment_spec.json`, and get the `fitness_score` from here too.


|fitness_score|mean_rewards_per_epi_stats_mean|mean_rewards_stats_mean|epi_stats_mean|solved_ratio_of_sessions|num_of_sessions|max_total_rewards_stats_mean|t_stats_mean|trial_id|variable_gamma|variable_hidden_layers|variable_lr|
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
|5.305994917071314|1.3264987292678285|195.404|154.2|1.0|5|200.0|199.0|dqn-2017_03_19_004714_t79|0.999|[64]|0.02|
|5.105207228739003|1.2763018071847507|195.13600000000002|160.6|1.0|5|200.0|199.0|dqn-2017_03_19_004714_t50|0.99|[32]|0.01|
|4.9561426920909355|1.2390356730227339|195.26000000000002|168.6|1.0|5|200.0|199.0|dqn-2017_03_19_004714_t78|0.999|[64]|0.01|
|4.76714626254895|1.1917865656372375|195.106|172.4|1.0|5|200.0|199.0|dqn-2017_03_19_004714_t71|0.999|[32]|0.02|
|4.717243567762263|1.1793108919405657|195.56400000000002|167.2|1.0|5|200.0|199.0|dqn-2017_03_19_004714_t28|0.97|[32]|0.001|

*`dqn-2017_03_19_004714_analysis_data.csv`, top 5 trials, from the [dqn-2017_03_19_004714](https://github.com/kengz/openai_lab/pull/73) experiment. We can see that among the dominating parameter values are gamma=0.999, hidden_layers=[64], lr=[0.02]. The best trial json below.*


```json
{
  "experiment_spec": {
    "Agent": "DQN",
    "HyperOptimizer": "GridSearch",
    "Memory": "LinearMemoryWithForgetting",
    "Optimizer": "AdamOptimizer",
    "Policy": "BoltzmannPolicy",
    "PreProcessor": "NoPreProcessor",
    "experiment_name": "dqn",
    "param": {
      "decay": 0,
      "exploration_anneal_episodes": 10,
      "gamma": 0.999,
      "hidden_layers": [64],
      "hidden_layers_activation": "sigmoid",
      "lr": 0.02
    },
    "problem": "CartPole-v0"
  },
  "metrics": {
    "epi_stats_mean": 154.2,
    "fitness_score": 5.305995,
    "max_total_rewards_stats_mean": 200,
    "mean_rewards_per_epi_stats_mean": 1.326499,
    "mean_rewards_stats_mean": 195.404,
    "solved_ratio_of_sessions": 1,
    "t_stats_mean": 199,
    "time_taken": "0:41:19"
  },
  "param_variables": {
    "gamma": 0.999,
    "hidden_layers": [64],
    "lr": 0.02
  },
  "stats": {
    "epi_stats": {
      "max": 195,
      "mean": 154.2,
      "min": 114,
      "std": 34.94796
    },
    "errored": false,
    "fitness_score": 5.305995,
  ...
```


This concludes the analysis. See the [solution PR here](https://github.com/kengz/openai_lab/pull/73). The best trial is `  dqn-2017_03_19_004714_t79`, with `fitness_score=5.305995`, and the variables:

- *lr*: 0.02
- *gamma*: 0.999
- *hidden_layers_shape*: [64]


Now that you know how to analyze the data,

- [start finding problems to beat and submitting your solutions](#solutions)
- [learn more about the evaluation metrics](#metrics)
