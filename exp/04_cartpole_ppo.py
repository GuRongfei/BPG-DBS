
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import gym
from BPG.ppo2 import PPO2
from BPG.ppo_policy import MlpPolicy

env_id = "CartPole-v1"
env = gym.make(env_id)
model = PPO2(MlpPolicy, env, learning_rate=1e-4, verbose=1, tensorboard_log="MLP/", full_tensorboard_log=True)
model.learn(int(10e3))
#model.save('../result/model/ppo_cartpole_2e5.pkl')
#model = model.load('../result/model/ppo_cartpole_2e5.pkl')

env = gym.make(env_id)
obs = env.reset()

for _ in range(10000):
    env.render()
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
env.close()

