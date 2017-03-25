# <a name="analysis"></a>Analysis

Once the Lab is running experiments, it will produce data. This section will details how to analyze and understand the data, before we can contribute to the [Best Solutions](#solutions).

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


We will illustrate with an example experiment data from the [Lab Demo](#demo).


## Graphs


### Session Graphs

When an experiment is running, the lab will draw the session graphs live, one for each session.

This graph has 3 subplots:

1. **total rewards and exploration rate vs episode**: directly records the total rewards attained (blue) at the end of each episode, in relation to the exploration rate (red), which could be `epsilon, tau`, etc. depending on the policy. The 2 lines usually show some negative correlation - when the exploration rate drops, the total rewards should rise. When a solution becomes stable, the blue line should stay around its max.

2. **mean rewards over the last 100 episodes vs episode**: measures the 100-episode mean of the total rewards from above. This metric is usually how a solution is identified - when it hits a target solution score, which would mean that the solution is *strong* and *stable* enough.

3. **loss vs time**: measures the loss of the agent's Neural Net. This graph is all the losses  concatenated over time, over all episodes. There is no definite unit for loss as it depends on what loss function we use in the NN architecture (typically `mean_squared_error`). As the NN starts getting more accurate, the loss should decrease.


<aside class="notice">
Use the session graph to immediately see how the agent is performing without needing to wait for the entire session to complete.
</aside>


<img alt="The best session graph" src="https://cloud.githubusercontent.com/assets/8209263/24180935/404370ea-0e8e-11e7-8f20-f8691ee03e7b.png" />

*The best session graph from the dqn experiment. From the session graph we can see that the agent starts learning the CartPole-v0 task at around episode 15, then solves and masters it before episode 20. Over time the loss decreases, the solution becomes stabler, and the mean rewards increases until the session is solved reliably.*


### Analysis Graph

The **analysis graph** is the primary visual used to judge the overall experiment - how all the trials perform. It is a pair-plots of the *measurement metrics on the y-axis*, and the *experiment variables on the x-axis*.


*(new adjacent possible)*

**The y-axis measurement metrics**

- `fitness_score`: the final evaluation metric the Lab uses to select a fit agent. The design and purpose of it is more involved - see [metrics](#metrics) for more.
- `mean_rewards_stats_mean`: the statistical mean of all the `mean_rewards` over all the sessions of a trial. Measures the average solution potential of a trial.
- `max_total_rewards_stats_mean`: the statistical mean of all the `max_total_rewards` over all the sessions of a trial. Measures the agent's maximum potential to gather rewards in an episode.
- `epi_stats_mean`: the statistical mean of the termination episode of a session. The lower the better, as it would imply that the session is solved faster.


**The hue**

Each data point represents a trial, with the relevant data point averaged over its sessions. The points are colored (see legend) with the hue:

- `solved_ratio_of_sessions`: how many sessions are solved out of the total sessions in a trial. 0 means no session is solved, 1 means all sessions are solved.

The granularity of the `solved_ratio_of_sessions` depends on the number of sessions ran per trial. From experience, we settle on 5 sessions per trial as it's the best tradeoff between granularity and computation time.


*(new adjacent possible)*

Multiple sessions allow us to observe the consistency of an agent. As we have noticed across the parameter space, there is a spectrum in terms of solvability: agents who cannot solve at all, can solve occasionally, can always solve. The agents in between may be valuable when developing an new algorithm, because a strong agent may be hard to find in the early stage.


how to read

what y are, hue, trial data points, how mean
high slow, closeness, corres to stability

swarmplot to show dist vs scatter plot

, over all trials. Do the mean mean. for each trial data take the mean.  The y ax

When an experiment is complete, the lab will aggregate the `trial_data` over all trials into `experiment_data`, and produce the experiment analysis data.




<div style="max-width: 100%"><img alt="The dqn experiment analytics" src="https://cloud.githubusercontent.com/assets/8209263/24087747/41a86170-0cf9-11e7-84b8-8f3fcae24c95.png" />
<br><br></div>
<img alt="The dqn experiment analytics correlation" src="https://cloud.githubusercontent.com/assets/8209263/24087746/418a9b54-0cf9-11e7-8aac-f0df817def43.png" />


_The dqn experiment analytics generated by the Lab. This is a pairplot, where we isolate each variable, flatten the others, plot each trial as a point. The darker the color the higher ratio of the repeated sessions the trial solves._


## Data

jsons and sorted CSV


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


## Metrics

Merge metrics and variables? later
