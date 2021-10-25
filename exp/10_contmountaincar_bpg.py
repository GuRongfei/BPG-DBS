import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import gym
from BPG.bpg import BPG
from BPG.policy import MlpPolicy

env_id = "MountainCarContinuous-v0"
env = gym.make(env_id)
model = BPG(MlpPolicy, env, learning_rate=1e-4, verbose=1)#, tensorboard_log="MLP/")
model.learn(int(10e2))
#model.save('../result/model/bpg_mcc_2e4.pkl')
#model = model.load('../result/model/bpg_mcc_2e4.pkl')

env = gym.make(env_id)
obs = env.reset()

for _ in range(10000):
    env.render()
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
env.close()

