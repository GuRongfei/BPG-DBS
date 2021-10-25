import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import gym
import fake_oscillator
from stable_baselines.ppo2 import PPO2
from stable_baselines.common.policies import MlpPolicy

import numpy as np

env_id = "fake_oscillator-v0"
env = gym.make(env_id)
model = PPO2(MlpPolicy, env, learning_rate=1e-5, verbose=1)#, tensorboard_log="MLP/")
#model.learn(int(1e5))
#model.save('../result/model/ppo_fake_1e5.pkl')
model = model.load('../result/model/ppo_fake_1e5.pkl')

env = gym.make(env_id)
obs = env.reset()

for _ in range(500):
    action = np.array([(np.random.rand()-0.5)*2])
    obs, rewards, dones, info = env.step(action)
    print('obs: ', obs)
    print("act: ", action)
    print('reward: ', rewards)

for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print('obs: ', obs)
    print("act: ", action)
    print('reward: ', rewards)

env.show('tst')
env.close()

