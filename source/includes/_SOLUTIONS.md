# <a name="solutions"></a>Best Solutions

Algorithms and best solutions by OpenAI Lab users. [Submission guideline, link to PR](http://kengz.me/openai_lab).


## Algorithms

The status of the algorithms in OpenAI Lab. Feel free to invent new ones!

_Pending: we still need a way to cross-evaluate algorithms. Refer to the NEC paper. Perhaps use Normalized human scores per episode vs Millions of Frames._

|algorithm|implemented?|paper|eval score|
|:---|:---|:---|:---|
|DQN|✓|[url](https://arxiv.org/abs/1312.5602)|-|
|double-DQN|✓|[url](https://arxiv.org/abs/1509.06461)|-|
|dueling-DQN|-|[url](https://arxiv.org/abs/1511.06581)|-|
|SARSA|✓|-|-|
|prioritized replay|in-progress|[url](https://arxiv.org/abs/1511.05952)|-|
|Q*(lambda)|-|-|-|
|Retrace(lambda)|-|-|-|
|DPG (Deterministic Policy Gradient aka actor-critic)|in-progress|[url](http://jmlr.org/proceedings/papers/v32/silver14.pdf)|-|
|DDPG (Deep-DPG, aka actor-critic with target networks)|in-progress|[url](https://arxiv.org/abs/1509.02971)|-|
|A3C (asynchronous advantage actor-critic)|-|[url](https://arxiv.org/pdf/1602.01783.pdf)|-|
|Neural Episodic Control (NEC)|next|[url](https://arxiv.org/abs/1703.01988)|-|
|EWC (Elastic Weight Consolidation)|-|[url](https://arxiv.org/abs/1612.00796)|-|


## Problems

If your algorithm beats the best solutions, please submit a [Pull Request](https://github.com/kengz/openai_lab/pulls) to the OpenAI Lab. Include the following:

- `<experiment_id>_analysis.png`
- `<experiment_id>_analysis_correlation.png`
- `<experiment_id>_analysis_data.csv`
- `<best_trial_id>.json`

See an [example PR here](https://github.com/kengz/openai_lab/pulls).


### Classic Problems

|problem|fitness score|epis before solve / best 100-epi mean|author|experiment_spec|
|:---|:---|:---|:---|:---|:---|
|CartPole-v0|5.3060|14|kengz/lgraesser|[dqn](http://kengz.me/openai_lab)|
|CartPole-v1|-|-|-|-|
|Acrobot-v1|-|-|-|-|
|MountainCar-v0|-|-|-|-|
|MountainCarContinuous-v0|-|-|-|-|
|Pendulum-v0|-|-|-|-|


### Box2D Problems

|problem|fitness score|epis before solve / best 100-epi mean|author|experiment_spec|
|:---|:---|:---|:---|:---|:---|
|LunarLander-v2|-|-|-|-|
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

