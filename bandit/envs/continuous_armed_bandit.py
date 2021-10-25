import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from gym import Env
from gym.spaces import Discrete, MultiDiscrete, MultiBinary, Box

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


class SYSTEM():
    def __init__(self, action_dim, fixed_matrix=True):
        self.beta = 0.9  ## randomness
        TARGET = -400
        self.target_action = np.ones(action_dim) * TARGET
        if (fixed_matrix):
            self.matrix = np.diag(np.ones(action_dim) * 0.1)
        else:
            # target_action = np.array([-100,-400,-400,200])
            A = np.random.rand(action_dim, action_dim)
            B = (A + A.T) / 2
            _, s, V = np.linalg.svd(B)
            c_matrix = np.zeros((action_dim, action_dim))
            np.fill_diagonal(c_matrix, np.ones(action_dim) * 0.1)
            B_new = V.T.dot(c_matrix).dot(V)
            self.matrix = B_new

    def step(self, action):
        u = np.random.rand()
        dis_vector = (action - self.target_action).reshape([-1, 1])
        cost = 0.001 * np.matmul(dis_vector.T, self.matrix).dot(dis_vector)
        cost = cost * self.beta + u * (1 - self.beta)
        reward = -cost
        return -reward


class ContBanditEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, action_dim=1, ep_length=10000):
        super(ContBanditEnv, self).__init__()
        self.action_dim = action_dim
        self.ep_length = ep_length

        self.beta = 0.9
        TARGET = 10
        self.target = TARGET
        self.target_action = np.ones(self.action_dim) * TARGET
        self.matrix = np.diag(np.ones(action_dim) * 0.1)

        # Dimensionality of our observation space
        self.action_space = Box(low=-1000, high=1000, shape=(self.action_dim,), dtype=np.float32)
        self.observation_space = Box(low=-1000, high=1000, shape=(1,), dtype=np.float32)

        # Episode Done?
        self.done = False
        self.current_step = 0

        # Our current state, with length(1,len_state)
        self.x = None

        # Reset environment
        self.reset()

    def step(self, action):
        self.current_step += 1
        self.done = self.current_step >= self.ep_length

        # Make vectorized form
        self.x += (action[0] - self.target)
        if abs(self.x[0])<0.2:
            self.x = [0.]
        elif self.x[0]>0.2:
            self.x = [1.]
        else:
            self.x = [-1.]
        #self.x = [-1.]

        arrayed_version = np.array(self.x)
        reward = self.Reward(arrayed_version, action)

        return arrayed_version, self.Reward(arrayed_version, action), self.done, {}

    def reset(self):
        """
        Reset environment, and get a window 250 of self.len_state size

        Returns:arrayed_version:np.array(1,len_state)

        """
        self.current_step = 0
        self.x = [0.]

        arrayed_version = np.array(self.x)

        return arrayed_version

    def render(self, mode='human', close=False):
        pass

    def show(self, title):
        pass

    def Reward(self, obs, action_value):
        u = np.random.rand()
        dis_vector = (obs + action_value - self.target_action).reshape([-1, 1])
        cost = np.matmul(dis_vector.T, self.matrix).dot(dis_vector)
        cost = cost * self.beta + u * (1 - self.beta)
        reward = -cost
        return reward






