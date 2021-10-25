import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from gym import Env
from gym.spaces import Discrete, MultiDiscrete, MultiBinary, Box

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


class FakeOscillatorEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, len_state=5, ep_length=10000):
        super(FakeOscillatorEnv, self).__init__()
        self.ep_length = ep_length

        # Dimensionality of our observation space
        self.dim = 1
        self.action_space = Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = Box(low=-1.5, high=1.5, shape=(len_state,), dtype=np.float32)

        # Episode Done?
        self.done = False
        self.current_step = 0

        # Our current state, with length(1,len_state)
        self.x = None
        self.x_state = []
        self.x_hist = []

        self.len_state = len_state

        # Reset environment
        self.reset()

    def step(self, action):
        val = float(action[0])
        self.x = max(min(1.5, self.x+val), -1.5)

        # Save our state
        self.x_state.append(self.x)
        self.x_hist.append(self.x)

        # Check length of our state
        if len(self.x_state) > self.len_state:
            self.x_state = self.x_state[1:]

        self.current_step += 1

        self.done = self.current_step >= self.ep_length

        # Make vectorized form
        arrayed_version = np.array(self.x_state)

        return arrayed_version, self.Reward(self.x, self.x_state, val), self.done, {}

    def reset(self):
        """
        Reset environment, and get a window 250 of self.len_state size

        Returns:arrayed_version:np.array(1,len_state)

        """
        self.current_step = 0
        self.x_state = []
        self.x_hist = []
        self.x = (np.random.rand(self.len_state)-0.5)*3

        for i in range(self.len_state):
            self.x = (np.random.rand() - 0.5) * 3
            self.x_state.append(self.x)

        arrayed_version = np.array(self.x_state)

        return arrayed_version

    def render(self, mode='human', close=False):
        pass

    def show(self, title):
        fig = plt.figure(figsize=(25, 12))
        ax = fig.add_subplot(111)
        ax.tick_params(labelsize=25)
        ax.plot(self.x_hist, '-', c='lightcoral', label='state')
        ax.legend(bbox_to_anchor=(1, 1), fontsize=25)
        ax.grid()

        ax.set_xlabel("TimeStep", fontsize=45, labelpad=15)
        ax.set_ylabel("state", fontsize=45, labelpad=15)
        ax.set_title(title, fontsize=55, pad=20)

        plt.savefig('../result/%s.png' % title)

    def Reward(self, x, x_state, action_value):
        """
        Super duper reward function, i am joking, just sum of absolute values which we supress + penalty for actions
        returns: float
        """
        """print("x: ", x)
        print("mean: ", np.mean(x_state))
        print("x_state: ", x_state)"""
        #return -(x - np.mean(x_state)) ** 2 - max(np.abs(1.5-x), np.abs(-1.5-x))# - 1. * np.abs(action_value)
        return -np.abs(np.mean(x_state))


