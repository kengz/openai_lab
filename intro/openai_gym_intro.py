import gym
env = gym.make('CartPole-v0')
env.reset()
# for _ in range(10):
#   env.render()
#   step = env.step(env.action_space.sample())
#   obs, reward, done, info = step

# for i_ep in range(5):
#   obs = env.reset()
#   for t in range(10):
#     env.render()
#     # print(obs)
#     action = env.action_space.sample()
#     print(action)
#     obs, reward, done, info = env.step(action)
#     if done:
#       print('ep finished')
#       break

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

from gym import spaces
space = spaces.Discrete(8)
print(space.sample())

from gym import envs
print(envs.registry.all())
print(envs.registry.make('CartPole-v0'))