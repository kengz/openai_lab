## <a name="metrics"></a>Metrics

(pending long writeup)

OpenAI Lab exists to address 2 major problems in RL, and WildML's Denny sums them up best in his post [Engineering Is The Bottleneck In (Deep Learning) Research](http://blog.dennybritz.com/2017/01/17/engineering-is-the-bottleneck-in-deep-learning-research/). They are:

**1. the difficulty of building upon other’s work**

As the Lab grows, we hope that engineers and researchers can experiment with an idea fast by building on top of our existing components.

**2. the lack of rigor in comparisons**

Multiple experiments running in the Lab will produce the same analytics and the evaluation metrics. This will allow us to compare algorithms and problems meaningfully, and that is the point of the Lab's [Solution Matrix](#solution-matrix).

We now describe the evaluation metrics for **problems** and **algorithms**.

### Problem Evaluation Metrics

problem x {algorithms} ~ solutions

fitness score on 4 parts:
stability and reproducibility (solve ratio),
speed (min episodes),
potential (max reward),
square for granularity

```
mean_rewards_per_epi * (1+solved_ratio_of_sessions)**2

ideal_mean_rewards_per_epi = mean_rewards / (epi/solved_epi_speedup)
ideal_solved_ratio = 1
ideal_fitness_score = fitness_score(
    ideal_mean_rewards_per_epi, ideal_solved_ratio)
return ideal_fitness_score
```

### Algorithm Evaluation Metrics

algorithm x {problems} ~ cross-solutions
