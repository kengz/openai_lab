# <a name="solutions"></a>Best Solutions

Algorithms and best solutions by OpenAI Lab users. We want people to start from working solutions instead of stumbling their ways there.

## Submission instructions

If you invent a new algorithm/combination that beats the best solutions, please submit a [Pull Request](https://github.com/kengz/openai_lab/pulls) to the OpenAI Lab.

To learn how to analyze experiment data, refer to [Analysis](#analysis).

Refer to the [PR template](https://github.com/kengz/openai_lab/blob/master/.github/PULL_REQUEST_TEMPLATE.md) for the submission guideline. See some previous example [solution PRs](https://github.com/kengz/openai_lab/pulls?q=is%3Apr+label%3Asolution+is%3Aclosed).


## <a name="solution-matrix"></a>Solution Matrix

A matrix of the best `fitness_score` of **Algorithms** v.s. **Problems**. The list of accepted solutions can be seen in the [solution PRs](https://github.com/kengz/openai_lab/pulls?q=is%3Apr+label%3Asolution+is%3Aclosed)

||DQN|double-DQN|SARSA|DDPG|
|:---|:---|:---|:---|:---|
|**CartPole-v0**|[9.487062](https://github.com/kengz/openai_lab/pull/73)|[10.31487](https://github.com/kengz/openai_lab/pull/78)|[13.41222](https://github.com/kengz/openai_lab/pull/91)|-|
|**CartPole-v1**|[11.68838](https://github.com/kengz/openai_lab/pull/80)|[14.12158](https://github.com/kengz/openai_lab/pull/82)|-|-|
|**Acrobot-v1**|-|-|-|-|
|**MountainCar-v0**|-|-|-|-|
|**MountainCarContinuous-v0**|-|-|-|-|
|**Pendulum-v0**|-|-|-|-|
|**LunarLander-v2**|[2.296613](https://github.com/kengz/openai_lab/pull/84)|[2.812674](https://github.com/kengz/openai_lab/pull/87)|-|-|
|**LunarLanderContinuous-v2**|-|-|-|-|
|**BipedalWalker-v2**|-|-|-|-|
|**BipedalWalkerHardcore-v2**|-|-|-|-|
|**CarRacing-v0**|-|-|-|-|
|**AirRaid-v0**|-|-|-|-|
|**Alien-v0**|-|-|-|-|
|**Assault-v0**|-|-|-|-|
|**Breakout-v0**|-|-|-|-|
|**MsPacman-v0**|-|-|-|-|
|**Pong-v0**|-|-|-|-|
|**Qbert-v0**|-|-|-|-|
|**SpaceInvader-v0**|-|-|-|-|
|**FlappyBird-v0**|-|-|-|-|
|**Snake-v0**|-|-|-|-|


## Algorithms

The status of the algorithms in OpenAI Lab. Feel free to invent new ones! For more detail on currently implemented algorithms, see [Algorithms](#algorithms)

_Pending: we still need a way to cross-evaluate algorithms. Refer to the NEC paper. Perhaps use Normalized human scores per episode vs Millions of Frames._

|algorithm|implemented?|eval score|
|:---|:---|:---|
|[DQN](https://arxiv.org/abs/1312.5602)|✓|-|
|[double-DQN](https://arxiv.org/abs/1509.06461)|✓|-|
|[dueling-DQN](https://arxiv.org/abs/1511.06581)|-|-|
|SARSA|✓|-|
|[prioritized replay](https://arxiv.org/abs/1511.05952)|in-progress|-|
|Q*(lambda)|-|-|
|Retrace(lambda)|-|-|
|[CEM (Cross Entropy Method)](https://en.wikipedia.org/wiki/Cross-entropy_method)|-|-|
|[PG (Policy Gradient)](https://webdocs.cs.ualberta.ca/~sutton/papers/SMSM-NIPS99.pdf)|-|-|
|[DPG (Deterministic Policy Gradient aka actor-critic)](http://jmlr.org/proceedings/papers/v32/silver14.pdf)|in-progress|-|
|[DDPG (Deep-DPG, aka actor-critic with target networks)](https://arxiv.org/abs/1509.02971)|in-progress|-|
|[A3C (asynchronous advantage actor-critic)](https://arxiv.org/pdf/1602.01783.pdf)|-|-|
|[TRPO](https://arxiv.org/abs/1502.05477)|-|-|
|[Neural Episodic Control (NEC)](https://arxiv.org/abs/1703.01988)|next|-|
|[EWC (Elastic Weight Consolidation)](https://arxiv.org/abs/1612.00796)|-|-|


## Problems

The list of accepted solutions can be seen in the [solution PRs](https://github.com/kengz/openai_lab/pulls?q=is%3Apr+label%3Asolution+is%3Aclosed)


### Classic Problems

|problem|fitness score|epis before solve / best 100-epi mean|author|experiment_spec|
|:---|:---|:---|:---|:---|:---|
|CartPole-v0|[13.41222](https://github.com/kengz/openai_lab/pull/91)|9|kengz/lgraesser|offpol_sarsa|
|CartPole-v1|[14.12158](https://github.com/kengz/openai_lab/pull/82)|16|kengz/lgraesser|double_dqn_v1|
|Acrobot-v1|-|-|-|-|
|MountainCar-v0|-|-|-|-|
|MountainCarContinuous-v0|-|-|-|-|
|Pendulum-v0|-|-|-|-|


### Box2D Problems

|problem|fitness score|epis before solve / best 100-epi mean|author|experiment_spec|
|:---|:---|:---|:---|:---|:---|
|LunarLander-v2|[2.812674](https://github.com/kengz/openai_lab/pull/87)|247|kengz/lgraesser|lunar_double_dqn|
|LunarLanderContinuous-v2|-|-|-|-|
|BipedalWalker-v2|-|-|-|-|
|BipedalWalkerHardcore-v2|-|-|-|-|
|CarRacing-v0|-|-|-|-|


### Atari Problems

|problem|fitness score|epis before solve / best 100-epi mean|author|experiment_spec|
|:---|:---|:---|:---|:---|:---|
|AirRaid-v0|-|-|-|-|
|Alien-v0|-|-|-|-|
|Assault-v0|-|-|-|-|
|Breakout-v0|-|-|-|-|
|MsPacman-v0|-|-|-|-|
|Pong-v0|-|-|-|-|
|Qbert-v0|-|-|-|-|
|SpaceInvader-v0|-|-|-|-|


### PyGame Problems

|problem|fitness score|epis before solve / best 100-epi mean|author|experiment_spec|
|:---|:---|:---|:---|:---|:---|
|FlappyBird-v0|-|-|-|-|
|Snake-v0|-|-|-|-|


### Universe Problems

|problem|fitness score|epis before solve / best 100-epi mean|author|experiment_spec|
|:---|:---|:---|:---|:---|:---|
|-|-|-|-|-|

