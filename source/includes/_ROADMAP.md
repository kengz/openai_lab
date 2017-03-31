# <a name="roadmap"></a>Roadmap

Check the latest under the [Github Projects](https://github.com/kengz/openai_lab/projects)


## <a name="motivations"></a>Motivations

We the authors never set out to build the OpenAI Lab with any grand vision in mind. We just wanted to test our RL ideas in the OpenAI Gym, faced many problems along the way, and their solutions became features. These opened up new adjacent possibles to do new things, and even more problems, and so on. Before we knew it, the critical components fell it place and we had something very similar to a scientific lab.

The problems faced by us are numerous and diverse, but there are several major categories. The first two are nicely described by WildML's Denny in his post [Engineering Is The Bottleneck In (Deep Learning) Research](http://blog.dennybritz.com/2017/01/17/engineering-is-the-bottleneck-in-deep-learning-research/), which resonates strongly with a lot of people.

**1. the difficulty of building upon other’s work**

If you have tried to implement any algorithms by looking at someone elses code, chances are it's painful. Sometimes you just want to research a small component like a prioritized memory, but you'd have to write 90% of the unrelated components from scratch. Simply look at the solution source codes submitted to the OpenAI Gym leaderboard; you can't extend them to build something much bigger.

There is no design or engineering standards for reinforcement learning, and that contributes to the major inertia in RL research. A lot of times research ideas are not difficult to come by, but implementing them is hard because there is *no reliable foundation to build on*.

We patiently built every piece of that foundation because giving up wasn't an option, so here it is. As the Lab grows, we hope that engineers and researchers can experiment with an idea fast by building on top of our existing components, and of course, contribute back.

**2. the lack of rigor in comparisons**

Denny describes this already, [read his blog](http://blog.dennybritz.com/2017/01/17/engineering-is-the-bottleneck-in-deep-learning-research/).

As the Lab became mature, we became more ambitious and try to solve more environment, with more agents. This naturally begs the question, "how do we compare them, across agents and environments?"

Multiple experiments running in the Lab will produce standardized data analytics and evaluation metrics. This will allow us to compare agents and environments meaningfully, and that is the point of the Lab's [Fitness Matrix](#fitness-matrix). It also inspired a [generalization of evaluation metrics](#metrics), which we have only discovered recently.

**3. inertia to high level vision**

When you're heels down implementing an algorithm and the extra 90% side components from scratch, it's hard to organize your work from a high level. Having to worry about other irrelevant components also makes you lose focus. The Lab removes that inertia and frees us from that myopic vision.

This freedom means more mental energy and time to focus on the essential components of research. It opens up new adjacent possibles and has us asking new questions. Below are a few things that we came up due to this freedom of high level vision:

- independently discovery of RankedMemory, which is essentially the Priorized Experience Replay (PER) idea.
- the study of hyperparameters at scale, with the [Analysis Graph](#analysis).
- new high level, powerful visuals from the [analysis graphs](#analysis) to eyeball the performance and potentials of an agent.
- auto-architecture of agent's neural net, because we wanted to study some variables of the NN architecture as hyperparameters.
- the perspective of RL as experimental science, and experimenting as breeding agents in the environments.
- the design of a new [`fitness_score` that provides richer evaluation metrics](#fitness), which considers solution speed, stability, consistency, and much more.
- the [Fitness Matrix](#fitness-matrix) and the ability to compare fitness_score across its grid.
- the [generalization of evaluation metrics](#generalization), which casts the hyperoptimization problem as one of temperature fields and differential calculus. Also gives the Fitness Fields Matrix.
