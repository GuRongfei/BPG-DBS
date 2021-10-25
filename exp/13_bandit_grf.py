import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

#import gym
#import bandit
#from BPG.grf_bpg import BPG
#from BPG.grf_policy import Actor, Critic
#import numpy as np

from utils.execute import Executor

env_id = 'continuous-armed_bandit-v0'
algo_id = 'ppo'
train_timestep = int(1e4)
test_timestep = 20
algo_para = {'learning_rate': 2.5e-3, 'gamma': 0, 'ent_coef': 0.0}

executor = Executor(env_id, algo_id, algo_para)
executor.train_policy(train_timestep)
executor.test_model(test_timestep)