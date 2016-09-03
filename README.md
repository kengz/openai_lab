# OpenAI Gym
Working out at the (OpenAI) gym

[OpenAI Gym Doc](https://gym.openai.com/docs) | [OpenAI Gym Github](https://github.com/openai/gym) | [RL intro](https://gym.openai.com/docs/rl)

# Installation

```shell
git clone https://github.com/kengz/openai_gym.git
cd openai_gym
# tensorflow cannot be via pypi
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/tensorflow-0.9.0-py3-none-any.whl
sudo pip3 install $TF_BINARY_URL
# install the rest
python setup.py install
```

It essentially performs the following (use only when you're resetting requirements.txt):

```shell
pip install git+https://github.com/openai/gym.git
pip install git+https://github.com/tflearn/tflearn.git
# generates requirements.txt
pip freeze > requirements.txt
```


# Roadmap

- use the pip install override, try. Fit all into `python setup.py install`
- try some simple algos on simple env (CartPole)