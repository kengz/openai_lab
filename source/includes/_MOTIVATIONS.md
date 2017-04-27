# <a name="motivations"></a>Motivations

*This section is more casual, but we thought we'd share the motivations behind the Lab.*

We the authors never set out to build OpenAI Lab with any grand vision in mind. We just wanted to test our RL ideas in OpenAI Gym, faced many problems along the way, and their solutions became features. These opened up new adjacent possibles to do new things, and even more problems, and so on. Before we knew it, the critical components fell it place and we had something very similar to a scientific lab.

The problems faced by us are numerous and diverse, but there are several major categories. The first two are nicely described by WildML's Denny in his post [Engineering Is The Bottleneck In (Deep Learning) Research](http://blog.dennybritz.com/2017/01/17/engineering-is-the-bottleneck-in-deep-learning-research/), which resonates strongly with a lot of people.

**1. the difficulty of building upon other's work**

If you have tried to implement any algorithms by looking at someone elses code, chances are it's painful. Sometimes you just want to research a small component like a prioritized memory, but you'd have to write 90% of the unrelated components from scratch. Simply look at the solution source codes submitted to the OpenAI Gym leaderboard; you can't extend them to build something much bigger.

Of many implementations we saw which solve OpenAI gym environments, many had to rewrite the same basic components instead of just the new components being researched. This is unnecessary and inefficient.

There is no design or engineering standards for reinforcement learning, and that contributes to the major inertia in RL research. A lot of times research ideas are not difficult to come by, but implementing them is hard because there is *no reliable foundation to build on*.

We patiently built every piece of that foundation because giving up wasn't an option, so here it is. As the Lab grows, we hope that engineers and researchers can experiment with an idea fast by building on top of our existing components, and of course, contribute back.

**2. the lack of rigor in comparisons**

Denny describes this already, [read his blog](http://blog.dennybritz.com/2017/01/17/engineering-is-the-bottleneck-in-deep-learning-research/).

As the Lab became mature, we became more ambitious and try to solve more environment, with more agents. This naturally begs the question, "how do we compare them, across agents and environments?"

Multiple experiments running in the Lab will produce standardized data analytics and evaluation metrics. This will allow us to compare agents and environments meaningfully, and that is the point of the Lab's [Fitness Matrix](#fitness-matrix). It also inspired a [generalization of evaluation metrics](#metrics), which we have only discovered recently.

**3. the inertia to high level vision**

When you're heels down implementing an algorithm and the extra 90% side components from scratch, it's hard to organize your work from a high level. Having to worry about other irrelevant components also makes you lose focus. The Lab removes that inertia and frees us from that myopic vision.

This freedom means more mental energy and time to focus on the essential components of research. It opens up new adjacent possibles and has us asking new questions.


**The New Adjacent Possibles**

With those problems above resolved, the Lab opens up the new adjacent possibles and allows us to do more. Below are some:

- independent discovery of RankedMemory, which is essentially the Priorized Experience Replay (PER) idea.
- the study of hyperparameters at scale, with the [Analysis Graph](#analysis).
- an explosion of experiments; suddenly we have the ability to test a lot of ideas quickly.
- new high level, powerful visuals from the [analysis graphs](#analysis) to eyeball the performance and potentials of an agent.
- auto-architecture of agent's neural net, because we wanted to study some variables of the NN architecture as hyperparameters.
- the perspective of RL as experimental science, and experimenting as breeding agents in the environments.
- the design of a new [`fitness_score` that provides richer evaluation metrics](#fitness), which considers solution speed, stability, consistency, and much more.
- the [Fitness Matrix](#fitness-matrix) and the ability to compare fitness_score across its grid.
- the [generalization of evaluation metrics](#generalization), which casts the hyperoptimization problem as one of temperature fields and differential calculus. Also gives the Fitness Fields Matrix.



## <a name="who-should-use"></a>Who Should Use OpenAI Lab?

We think this framework is useful for two types of users, those who are new to RL and RL researchers / advanced RL users.


### Newcomers to RL

For users that are new to RL, there is a lot to get your head around before you can get started. Understanding the Open AI gym environments and how to work with them, understanding RL algorithms, and understanding neural networks and their role as function approximators. OpenAI Lab reduces the inertia to begin.

We provide a range of implemented algorithms and components, as well as solutions to OpenAI gym environments that can be used out of the box. We also wanted to give new users the flexibility to change as much as possible in their experiments so we parameterized many of the algorithm variables and exposed them through a simple JSON interface.

This makes it possible for new users to spend their time experimenting with the different components and parameter settings, understanding what they do and how they affect an agent's performance, instead of going through the time consuming process of having to implement things themselves. 


### RL Researchers / Advanced Users

For advanced users or RL researchers, the Lab makes it easier to develop new ideas (a new way of sampling from memory for example) because a user needs to write only that code and can reuse the other components.

It also makes it easier to build on other peoples work. Solutions submitted via the lab's framework encapsulate all of the code and parameter settings, so that the result can be reproduced in just a few lines of code.

Additionally, the experiment framework and analytics allows better measurement of their ideas, and standardization provides meaningful comparisons among algorithms.


## <a name="main-contributions"></a>Main Contributions

We see the main new contributions of the Lab as follows:

1. It is the first example of a framework to do deep RL that we know of that provides both a simple and unified interface to the OpenAI gym and neural network libraries which are necessary to implement and run RL agents, and out of the box components and algorithms. 
2. The reusability of components.
3. Framework for running hundreds of trials searching multidimensional parameter space in just a few lines of code.
4. Automated analytics across all of these experiments making it easier to understand what is happening quickly.
5. The [Fitness matrix](#fitness-matrix).

1 - 4 have been covered in the sections above. Here we'll focus on the [Fitness matrix](#fitness-matrix)

The Fitness Matrix compares algorithms and environments. Tables playing a similar role to this are often seen in RL research papers from DeepMind or OpenAI.

Clearly it would take a lot for an individual to produce something similar - first to implement so many algorithms, then to run them across many environments, which would consume a lot of computational time.

We think the Fitness Matrix could be useful for RL researchers or new users as a benchmark to evaluate their new algorithms. An open source fitness matrix means everyone can use it instead of building one from scratch.

We see this as an extension of OpenAI gym's Evaluation boards ([example: CartPole-v0](https://gym.openai.com/envs/CartPole-v0)) in two ways:
- First, the score is richer; it takes into account not just the speed to solve a problem and the maximum score, but also the stability and consistency of solutions. See [Fitness Score](#fitness).
- Second, it makes comparison of environments and the comparison of agents across environments possible.

Our long term goal is to build a community of users and contributors around this framework. The more contributors, the more advanced the algorithms that are implemented and comparable wth the Lab. If successful, it could serve as the standardized benchmark for a large class of algorithms and a large class of problems. It becomes a public, living record which can be used by researchers who need to evaluate new algorithms and have no access to detailed benchmarks (unless if you're Google/OpenAI). 

With these metrics across multiple environments and agents, we can characterize, say, if an agent is strong (solves many problems); or if a generic component like Prioritized Memory Replay can increase the score of agent across all environments.

We also think the Fitness matrix could be used to further research, e.g. let us study cross-environment evaluation metrics, find patterns, and classify the characteristic of problems and agents. For example, we noticed that some classes of problems require spacial reasoning, or multi-step learning to solve; and there are agents who could or could not do that.

For the formulation of these measurements and more details, see [generalized metrics](#generalized-metrics). Such high level, cross-environment / agent analysis seems fairly new, and could prove useful to researchers. We would like to continue our work on this.

