import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import gym
import gym_oscillator
import oscillator_cpp
from stable_baselines.common import set_global_seeds

from BPG.policy import MlpPolicy,MlpLnLstmPolicy,FeedForwardPolicy
from stable_baselines.common.vec_env import DummyVecEnv,SubprocVecEnv,VecNormalize, VecEnv
from BPG.bpg import PPO2
from stable_baselines.common.vec_env import VecEnv

import datetime

import xlwt
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

env_id = 'oscillator-v0'
time_step = int(10e3)

def make_env(env_id, rank, seed=0, ):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :param s_i: (bool) reward form, only one can be true
    """

    def _init():
        env = gym.make(env_id)
        print(env.reset().shape)
        return env

    set_global_seeds(seed)
    return _init


def gen_model():
    num_cpu = 1
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    model = PPO2(MlpPolicy, env, verbose=1)#, tensorboard_log="MLP/")
    model.learn(time_step)
    return model


def test(model):
    env = gym.make(env_id)
    obs = env.reset()
    infos = []#rews_, obs_, acs_, states_x, states_y

    for i in range(5000):
        obs, rewards, dones, info = env.step([0])
        infos.append([rewards, obs[0], 0, env.x_val, env.y_val])

    for i in range(10000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        infos.append([rewards, obs[0], action, env.x_val, env.y_val])

    for i in range(2000):
        obs, rewards, dones, info = env.step([0])
        infos.append([rewards, obs[0], 0, env.x_val, env.y_val])

    infos = np.array(infos)
    s = infos[:, 3]
    a = infos[:, 2]
    plot_sa(s, a, "0616tst")


def plot_sa(s, a, title):
    fig = plt.figure(figsize=(25, 12))
    ax = fig.add_subplot(111)
    ax.tick_params(labelsize=25)
    ax.plot(a, '-', c='lightcoral', label='Action')
    ax2 = ax.twinx()
    ax2.tick_params(labelsize=25)
    ax2.plot(s, '-r', c='steelblue', label='State ')
    ax.legend(bbox_to_anchor=(1, 1), fontsize=25)
    ax.grid()

    ax.set_xlabel("TimeStep", fontsize=45, labelpad=15)
    ax.set_ylabel("Actions", fontsize=45, labelpad=15)
    ax2.set_ylabel("States", fontsize=45, labelpad=15)
    ax2.legend(bbox_to_anchor=(0.85, 1), fontsize=25)
    ax.set_title(title, fontsize=55, pad=20)

    plt.savefig('../result/%s.png' % title)


if __name__ == "__main__":
    model = gen_model()
    test(model)
