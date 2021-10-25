
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import gym
from stable_baselines.deepq import DQN

env_id = "CartPole-v1"
env = gym.make(env_id)
model = DQN('MlpPolicy', env, learning_rate=1e-4, verbose=1)#, tensorboard_log="MLP/")
#model.learn(int(2e5))
#model.save('../result/model/dqn_cartpole_2e5.pkl')
model = model.load('../result/model/dqn_cartpole_10e5.pkl')

env = gym.make(env_id)
obs = env.reset()

for _ in range(10000):
    env.render()
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
env.close()

