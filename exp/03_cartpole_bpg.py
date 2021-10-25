
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import gym
from BPG.bpg import BPG
from BPG.policy import MlpPolicy

env_id = "CartPole-v1"
env = gym.make(env_id)
model = BPG(MlpPolicy, env, learning_rate=1e-4, verbose=1)#, tensorboard_log="MLP/")
model.learn(int(2e5))
model.save('../result/model/bpg_cartpole_2e5.pkl')
#model = model.load('../result/model/bpg_cartpole_10e5.pkl')

env = gym.make(env_id)
obs = env.reset()

for _ in range(10000):
    env.render()
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
env.close()

