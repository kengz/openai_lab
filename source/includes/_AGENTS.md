# <a name="agents"></a>Agents

Agents are containers for reinforcement learning algorithms. They consist of a number of differnet components which are specified in the experiment_specs.

- `Agent` (Learning algorithm): decision function for learning from experiences gained by acting in an environment (eg Q-Learning, Sarsa). This is also the main class for agents. All other components of an agent are contained within this class.
- `Policy`: decision function for acting in an environment. Controls exploration vs. exploitation trade off(e.g. epsilon greedy, boltzmann)
- `Memory`: for storing experiences gained by acting in an environment. Controls how experiences are sampled for an agent to learn from. (e.g. random uniform with no forgetting, prioritized sampling with forgetting)
- `Optimizer`: controls how to optimize the function approximators contained within the agent (e.g. Stochatic Gradient Descent, Adam) 
- `HyperOptimizer`: hyperparameter optimization algorithms used to vary the agent parameters and run trials with them (e.g grid search, random search)
- `Preprocessor`: controls the transformations made to state representaions before being passed as inputs to the policy and learning algorithm. (e.g. no preprocessing, concatenating current and previous state). Useful for Atari.

To define an `Agent` you must specify each of the components. The example below is from the specification for `dqn` in `rl/spec/classic_experiment_specs.json`. The `rl/spec/*.json` files contains lots of other examples.

```json
"Agent": "DQN",
"HyperOptimizer": "GridSearch",
"Memory": "LinearMemoryWithForgetting",
"Optimizer": "AdamOptimizer",
"Policy": "BoltzmannPolicy",
"PreProcessor": "NoPreProcessor",
```

Each of the components with the exception of `Agent` and `Policy` are uncoupled, so can be freely switched in an out for different types of components. Different combinations of components may work better than others. We leave that up to you to experiment with. For some inspiration, see [our best solutions](#solutions)

For the currently implemented algorithms, the following `Agents` can go with the following `Policies`.

- DQN, Sarsa, ExpectedSarsa, OffPolSarsa, FreezeDQN: EpsilonGreedyPolicy, OscillatingEpsilonGreedyPolicy, TargetedEpsilonGreedyPolicy, BoltzmannPolicy
- DoubleDQN: DoubleDQNPolicy, DoubleDQNBoltzmannPolicy

## Agent

See [algorithms secton](#algorithms) for an explanation of currently implemented agents (learning algorithms). Agents always need the following parameters: `gamma` (how much to discount the future), `hidden_layers` (list of hidden layer sizes for neura network function approximators), and `hidden_layers_activation`,  (activation function for the hidden layers). The input and output layer sizes are inferred from the environment specs.

```json
"dqn": {
    "Agent": "DQN",
    "param": {
      "hidden_layers": [32, 16],
      "hidden_layers_activation": "sigmoid",
    },
```

To make used of the random search hyperoptimizer for the network architecture, it is necessary to use the *auto-architecture* mode for building the network.  In this case set the `auto-architecture` param to true, and specific the `num_hidden_layers ` and the `first_hidden_layer_size`.

```json
"dqn": {
    "Agent": "DQN",
    "param": {
      "hidden_layers_activation": "sigmoid",
      "auto_architecture": true,
      "num_hidden_layers": 3,
      "first_hidden_layer_size": 512
    },
```

## Policy

A policy is a decision function for acting in an environment. Policies take as input a description of the state space and output an action for the agent to take.

Depending on the algorithm  used, agents may directly approximate the policy (policy based algorithms) or have an indirect policy, that depends on the Q-value function approximation (value based algorithms). Algorithms that approximate both the policy and the Q-value function are known as actor-critic algorithms.

All of the algorithms implemented so far are value-based. The policy for acting at each timestep is often a simple epsilon-greedy policy.

![](./images/e_greedy.png)

Alternatively, an indirect policy may use the Q-value to output a probability distribution over actions, and sample actions based on this distribution. This is the approach taken by the Boltzmann policies.

A critical component of the policy is how is balances *exploration* vs. *exploitation*. To learn how to act well in an environment an agent must *explore* the state space.  The more random the actions an agent takes, the more it explores. However, to do well in an environment, an agent needs to take the best possible action given the state. It must *exploit* what it has learnt.

Below is a summary of the currently implemented policies. Each takes a slightly different approach to balancing the exploration-exploitation problem.

### EpsilonGreedyPolicy
Parameterized by starting value for epsilon (`init_e`), min value for epsilon (`final_e`), and the number of epsiodes to anneal epsilon over (`exploration_anneal_episodes`). The value of epsilon is decayed linearly from start to min.

```json
"dqn": {
    "Policy": "EpsilonGreedyPolicy",
    "param": {
      "init_e" : 1.0,
      "final_e" : 0.1,
      "exploration_anneal_episodes": 100
    },
```

### DoubleDQNPolicy
When actions are not random this policy selects actions by summing the outputs from each of the two Q-state approximators before taking the max of the result. Same approach as EpsilonGreedyPolicy to decaying epsilon and same params.

### BoltzmannPolicy
Parameterized by the starting value for tau (`init_tau`), min value for tau (`final_tau`), and the number of epsiodes to anneal epsilon over (`exploration_anneal_episodes`). At each step this policy selects actions based on the following probability distribution

![](./images/boltzmann.png)

Tau is decayed linearly over time in the same way as in the EpsilonGreedyPolicy.

```json
"dqn": {
    "Policy": "BoltzmannPolicy",
    "param": {
      "init_tau" : 1.0,
      "final_tau" : 0.1,
      "exploration_anneal_episodes": 10
    },
```

### DoubleDQNBoltzmannPolicy
Same as the Boltzmann policy except that the Q value used for a given action is the sum of the outputs from each of the two Q-state approximators.

### TargetedEpsilonGreedyPolicy
Same params as epsilon greedy policy. This policy swtches between active and inactive exploration cycles controlled by partial mean rewards and it distance to the target mean rewards.

### DecayingEpsilonGreedyPolicy
Same params as epsilon greedy policy. Epsilon is decayed exponentially.

### OscillaitngEpsilonGreedyPolicy
Same as epsilon greedy policy except at episode 18 epsilon is dropped to the max of 1/3 or its current value or min epsilon.

### Creating your own

A policy has to have the following functions. You can create your own by inheriting from Policy or one of its children.

```python
def select_action(self, state):
        '''Returns the action selected given the state'''

def update(self, sys_vars):
    '''Update value of policy params (e.g. epsilon)
    Called each timestep within an episode'''
```

## Memory

The agent's memory stores experiences that an agent gains by acting within an environment. An environment is in a particular state.  Then the agent acts, and receives a reward from the environment. The agent also receives information about the next state, including a flag indicating whether the next state is the terminal state. Finally, an error measure is stored, indicating how well an agent can estimate the value of this particular transition. 

This information about a single step is stored as an experience. Each experience consists  of

- Current state
- Action taken
- Reward
- Next state
- Terminal
- Error

```python
(state, action, nextstate, reward, terminal, error)
```

Crucially, the memory controls how long experiences are stored for, and which experiences are sampled from it to use as input into the learning algorithm of an agent. Below is a summary of the currently implemented memories

### LinearMemory
The size of the memory is unbounded and experiences are sampled random uniformly from memory.

### LinearMemoryWithForgetting

```json
"dqn": {
    "Memory": "LinearMemoryWithForgetting",
    "param": {
      "max_len" : 10000
    },
```

Parameterizes by `max_len` param which bounds the size of the memory. Once memory reaches the max size, the oldest experiences are deleted from the memory to make space for new experiences. Experiences are sampled random uniformly from memory.

### LeftTailMemory
Like linear memory with sampling via a left-tail distribution. This has the effect of drawing more from newer experiences.

### PrioritizedExperienceReplay
Experiences are weighted by the error, a measure of how well the learning algorithm currently performs on that experience. Experiences are sampled from memory in proportion to the p value (adjusted error value)

```python
p = (1 + e)** alpha
```

The parameter `e` > 0 is  a constant added onto the error to prevent experiences with error 0 never being sampled. `alpha` controls how spiked the distribution is. The lower `alpha` the closer to unform the distribution is. `alpha` = 0 corresponds to uniform random sampling.

```json
"dqn": {
    "Memory": "PrioritizedExperienceReplay",
    "param": {
      "e" : 0.01,
      "alpha": 0.6,
      "max_len" : 10000
    },
```

This has the effect of drawing more from experiences that the learning algorithm doesn't perform well on, i.e. the experiences from which is has most to learn. The size of the memory is bounded as in LinearMemoryWithForgetting.

### RankedMemory

### Creating your own

A memory has to have the following functions. You can create your own by inheriting from Memory or one of its children.

```python
def add_exp(self, action, reward, next_state, terminal, error):
    '''add an experience to memory'''

def get_exp(self, inds):
    '''get a batch of experiences by indices
       helper function called by rand_minibatch'''

def pop(self):
    '''get the last experience (batched like get_exp()'''

def size(self):
    '''returns the size of the memory'''

def rand_minibatch(self, size):
        '''returns a batch of experiences sampled from memory'''
```

## Optimizer
Controls how to optimize the function approximators contained within the agent. For feedforward and convolutional neural networks, we suggest using Adam with the default parameters for everything except the learning rate as this is widely considered to be the best algorithm for optmizing deep neural network based function approximators. For recurrent neural networks we suggest using RMSprop.

### SGD
Stochastic Gradient Descent. Parameterized by `lr` (learning rate), `momentum`, `decay` and `nestorov. See [Keras](https://keras.io/optimizers/#sgd) for more details.

```json
 "dqn": {
    "Optimizer": "SGDOptimizer",
    "param": {
      "lr": 0.02,
      "momentum" : 0.9
      "decay": 0.00001
      "nesterov": true
    },
```

### Adam
Parameterized by `lr` (learning rate),  `beta_1`, `beta_2`, `epsilon`, `decay`. See [Keras](https://keras.io/optimizers/#adam) for more details.

### RMSprop
Parameterized by `lr` (learning rate),  `rho`, `epsilon`, `decay`. See [Keras](https://keras.io/optimizers/#rmsprop) for more details.

## HyperOptimizer

Controls how to search over your hyperparameter space. We suggest using each of the three hyperoptimizers in the following order when trrying to find the optimal parameters for an agent in an environment as these correspond to rough grained >> finer grained search

1. LineSearch
2. GridSearch
3. RandomSearch

### LineSearch
### GridSearch
### RandomSearch


### HyperOptimizer Roadmap

These are the future hyperparameter optimization algorithms we'd like to implement standalone in the Lab. The implementations for them currently exists, but they're too bloated, and their engineering aspects are not ideal for the Lab.

- [TPE](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf) / [hyperopt](https://github.com/hyperopt/hyperopt)
- [Bayesian Optimizer (Spearmint)](https://github.com/HIPS/Spearmint)
- [SMAC](http://www.cs.ubc.ca/labs/beta/Projects/SMAC/#software)


### Implementation Guideline

All implemented hyperoptimizers shall extend the base `HyperOptimizer` class in `rl/hyperoptimizer/base_hyperoptimizer.py` and follow its design for compatibility. Below we show this design to be general theoretically and practically. Moreover, do not use bloated dependencies.

**Theoretical design:**

A hyperoptimizer is a function `h` that takes:

- a trial (objective) function `Trial`
- a parameter space `P` (implemented in `experiment_spec`)

and runs the algorithm:

1. search the next `p` in `P` using its internal search algorithm, add to its internal `param_search_list`.
2. run a (slow) function `Trial(p) = fitness_score` (inside trial data)
3. update search using the feedback `fitness_score`
4. repeat until max steps or fitness condition met

Note that the search space `P` is a tensor space product of `m` bounded real spaces `R` and `n` bounded discrete spaces `N`. The search path in `param_search_list` must also be well-ordered to ensure resumability.


**Implementation requirements:**

1. we want order-preserving and persistence in search for the ability to resume/reproduce an experiment.
2. the search algorithm may have its own internal memory/belief to facilitate search.
3. the `Trial` function shall be treated as a blackbox `Trial(p) = fitness_score` with input/output `(p, fitness_score)` for the generality of implementation/


**Specification of search space:**

1\. for real variable, specify a distribution (an interval is just a uniformly distributed space) in the `experiment_spec.param_range`. Example:

```json
"lr": {
  "min": 0.0005,
  "max": 0.05
}
```

2\. for discrete variable, specify a list of the values to search over (since it is finite anyway) in the `experiment_spec.param_range`. This will automatically be sorted when read into `HyperOptimizer` to ensure ordering. Example:


```json
"lr": [0.01, 0.02, 0.05, 0.1, 0.2]
```


The hyperopt implementation shall be able to take these 2 types of param_range specs and construct its search space.

Note that whether a variable is real or discrete can be up to the user; some variable such as `lr` can be sampled from interval `0.001 to 0.1` or human-specified options `[0.01, 0.02, 0.05, 0.1, 0.2]`. One way may be more efficient than the other depending on the search algorithm.

The experiment will run it as:

```python
# specify which hyperoptimizer class to use in spec for bookkeeping
Hopt = get_module(GREF, experiment_spec['HyperOptimizer'])
hopt = Hopt(Trial, **experiment_kwargs)
experiment_data = hopt.run()
```

## Preprocessor

Sometimes preprocessing the states before they are received by the agent can help to simplify the problem or make the agent strong. One example is the pixel preprocessing the removes color channels and rescales image size, in order to reduce unnecessary information overload. The other is to concat states from sequential timesteps to present richer, correlated information that is otherwise sparse.

The change in dimensions after preprocessing is handled automatically, so you can use them without any concerns.

### NoPreProcessor

The default that does not preprocess, but pass on the states as is.

### StackStates

Concat the current and the previous states. Turns out this boosts agent performance in the `LunarLander-v2` environment.

### DiffStates

Take the difference `new_states - old_states`.

### Atari

Convert images to greyscale, downsize, crop, then stack 4 most recent states together. Useful for the Atari environments.


## problem

Problems are not part of agent, but they are part of the `experiment_spec` that gets specified with the agent.

We have not added all the OpenAI gym environments AKA problems. If you get to new environments using the lab, please add them in `rl/spec/problems.json`, and it should be clear from those examples.

Moreover, when adding new problems, consider the dependencies setup too, such as Mujoco. Please add these to the `bin/setup` so other users could run it.
