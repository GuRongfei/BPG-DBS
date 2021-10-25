import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import gym
import gym_oscillator
import oscillator_cpp
from BPG.bpg import BPG
from BPG.policy import MlpPolicy

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

env_id = "oscillator-v0"
env = gym.make(env_id)
model = BPG(MlpPolicy, env, learning_rate=1e-4, verbose=1)#, tensorboard_log="MLP/")
model.learn(int(1e4))
#model.save('../result/model/bpg_dbs_10e6.pkl')


def plot_reward(data, name):
    fig = plt.figure(figsize=(25, 12))
    ax = fig.add_subplot(111)
    ax.tick_params(labelsize=25)
    ax.plot(data, '-', c='lightcoral', label=name)
    ax.legend(bbox_to_anchor=(1, 1), fontsize=25)
    ax.grid()

    ax.set_xlabel("TimeStep", fontsize=45, labelpad=15)
    ax.set_ylabel(name, fontsize=45, labelpad=15)
    ax.set_title(name, fontsize=55, pad=20)

    plt.savefig('../result/%s.png' % name)

#model = model.load('../result/model/bpg_dbs_10e6.pkl')

env = gym.make(env_id)
obs = env.reset()


def test(model):
    env = gym.make(env_id)
    obs = env.reset()
    infos = []#rews_, obs_, acs_, states_x, states_y

    for i in range(2000):
        obs, rewards, dones, info = env.step([0])
        infos.append([rewards, obs[0], 0, env.x_val, env.y_val])

    for i in range(5000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        infos.append([rewards, obs[0], action, env.x_val, env.y_val])

    for i in range(1000):
        obs, rewards, dones, info = env.step([0])
        infos.append([rewards, obs[0], 0, env.x_val, env.y_val])

    infos = np.array(infos)
    s = infos[:, 3]
    a = infos[:, 2]
    plot_sa(s, a, "0830tst")


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


test(model)