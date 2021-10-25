import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import gym
from stable_baselines.ppo2 import PPO2
from stable_baselines.common.policies import MlpPolicy

env_id = "MountainCarContinuous-v0"
env = gym.make(env_id)
model = PPO2(MlpPolicy, env, learning_rate=1e-4, verbose=1)#, tensorboard_log="MLP/")
#model.learn(int(2e4))
#model.save('../result/model/ppo_mcc_2e4.pkl')
model = model.load('../result/model/ppo_mcc_2e4.pkl')

env = gym.make(env_id)
obs = env.reset()

for _ in range(10000):
    env.render()
    action, _states = model.predict(obs)
    #tmp = [0, 0]
    #tmp[0], tmp[1] = obs[0], obs[1]
    #tmp[0] += 0.57893821
    print('obs: ', obs)
    print("act: ", action)
    obs, rewards, dones, info = env.step(action)
env.close()

