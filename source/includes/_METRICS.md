# <a name="metrics"></a>Metrics


Running experiments at this scale necessitates a higher level metric.

The lab makes comparisons among agents and environments possible - it's almost screaming for it. NOPE. In the Lab, we are breeding hundreds of agents across many environments. The choice of the name is intentional

With the Lab, we are breeding hundreds of agents across many environments, and selecting the best ones. Naturally, this selection metric the `fitness_score`. Some evolutionary search algorithm for the `HyperOptimizer` is on our roadmap.

Also briefly talk about the higher level generalized metrics, with fitness_score as the temperature function.


## Fitness Score

The fitness score of a trial is defined as

`fitness_score = mean_rewards_per_epi_mean * [(1+stability_mean) * ((1+consistency)**2)] ** sign(mean_rewards_per_epi_mean)`

or renaming variables by what the terms represent:

`fitness_score = power * distinguisher`

where

- `power = mean_rewards_per_epi_mean`
- `distinguisher = amplifier ** sign(power)`
- `amplifier = (1+stability_mean) * ((1+consistency)**2)`

The fitness score is designed to capture the following:

1. **strength**: `mean_rewards`
2. **speed**: `1/epi`
3. **stability**: session-level stability, `stability_gap / mastery_gap`
4. **consistency**: AKA trial-level stability, `solved_ratio_of_session`
5. **granularity**: in `1+stability` and `1+consistency` to ensure partial solution doesn't get lost when stability and consistency = 0
6. **amplification**: amplify by session-level stability linearly with `*(1+stability)`, and by trial-level stability quadratically with `*(1+consistency)**2`
7. **distinguishability**: multiply amplifier if `power` is positive, else divide; essentially `*(amplifier**sign(power))`


### Strength

The higher (more positive) the `mean_rewards` an agent gets, the stronger it is.

### Speed

Given two same `mean_rewards`, the agent that achieves it in less episodes is faster, with `speed = 1/epi`. This yields the notion of `power = strength * speed = mean_rewards_per_epi`. Use the sessions-mean of a trial, i.e. `mean_rewards_per_epi_mean` = mean of multiple `mean_rewards_per_epi` values.

### Stability

`stability = stability_gap / mastery_gap` for a session, with range `0.0 - 1.0` from unstable to stable. Measures session-level stability. Use the sessions-mean of a trial, i.e. mean of multiple `stability` values.

- `stability_gap` = the episodic-length of measurement, typically 100 episodes, given in `problem['REWARD_MEAN_LEN']`
- `mastery_gap` = how many episodes since the first solution does it take to solve the environment

`mastery_gap = max(last_epi - first_solved_epi, stability_gap)`, so that we use `mastery_gap = stability_gap` for any faster mastery to cap the ratio at 1.0. If problem is unsolved, set `mastery_gap = INF` to yield stability 0.0.

As for determining `first_solved_epi`, for a solvable problem, it's the first index (episode) of when `total_rewards > problem['SOLVED_MEAN_REWARD']`; for an unsolved problem (unbounded rewards) the criteria changes to when `total_rewards > 0.95 * max_total_rewards`, with 0.95 chosen for 1 sigma variation.

### Consistency

`consistency = solved_ratio_of_session` for a trial, with range `0.0 - 1.0` from inconsistent to consistent. This is the trial-level measurement of stability, as it measures how consistently the agent can solve an environment given multiple repeated sessions.

`consistency = 0` always for unsolved problems (unbounded rewards) since solution is undefined.

### Granularity

When `stability=0` or `consistency=0` the multiplier will drop to zero, regardless if there is any partial solutions. This will make training harder as the agent will not learn from partial solutions if these are treated as non-solutions. So, restore the granularity that preserves partial solution simply by adding 1, i.e. `1+stability` and `1+consistency`.

### Amplification

To separate solutions from noise, amplify the good ones and separate them out, while diminish and cluster the worse ones together. Amplify by session-level stability linearly with `*(1+stability)` since it's of the same order as `power`. Amplify by trial-level stability quadratically with `*(1+consistency)**2` since trial stability is of the next order. Amplifier is always positive.

### Distinguishability

Always amplify to make better solutions have more positive fitness score. If `power` is negative, amplify toward the 0 axis, i.e. divide by amplifier. If `power` is positive, amplify away from the 0 axis, i.e. multiply the amplifier. Essentially, `distinguisher = amplifier**sign(power)`.


## Motivations

OpenAI Lab exists to address 2 major problems in RL, and WildML's Denny sums them up best in his post [Engineering Is The Bottleneck In (Deep Learning) Research](http://blog.dennybritz.com/2017/01/17/engineering-is-the-bottleneck-in-deep-learning-research/). They are:

**1. the difficulty of building upon other’s work**

As the Lab grows, we hope that engineers and researchers can experiment with an idea fast by building on top of our existing components.

**2. the lack of rigor in comparisons**

Multiple experiments running in the Lab will produce the same analytics and the evaluation metrics. This will allow us to compare algorithms and problems meaningfully, and that is the point of the Lab's [Solution Matrix](#solution-matrix).

We now describe the evaluation metrics for **problems** and **algorithms**.

### Problem Evaluation Metrics

problem x {algorithms} ~ solutions


### Algorithm Evaluation Metrics

algorithm x {problems} ~ cross-solutions
