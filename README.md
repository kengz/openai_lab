# OpenAI Gym
Working out at the (OpenAI) gym

[OpenAI Gym Doc](https://gym.openai.com/docs) | [OpenAI Gym Github](https://github.com/openai/gym) | [RL intro](https://gym.openai.com/docs/rl)

## Installation

```shell
git clone https://github.com/kengz/openai_gym.git
cd openai_gym
python setup.py install
```

It essentially performs the following (use only when you're resetting requirements.txt). 

Additionally, you may want to install the correct distribution of Tensorflow yourself.

```shell
pip3 install h5py numpy scipy pandas matplotlib
# choose the right distribution for ur machine
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.10.0rc0-py3-none-any.whl
sudo pip3 install $TF_BINARY_URL
pip3 install git+https://github.com/tflearn/tflearn.git
pip3 install keras
pip3 install git+https://github.com/openai/gym.git
# generates requirements.txt
pip3 freeze > requirements.txt
```


## Usage

Run each file in the folders.

To view **Tensorboard**, do `tensorboard --logdir='/tmp/tflearn_logs/'`


## Roadmap

- try some simple algos on simple env (CartPole)