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

env_id = 'discrete-osc-v0'
algo_id = 'ppo'
train_timestep = int(5e5)
test_timestep = 2000
env_para = {'len_state': 250}
algo_para = {'learning_rate': 2.5e-4, 'gamma': 0.99, 'ent_coef': 0.01, 'n_steps': 128, 'lam': 0.95}

executor = Executor(env_id, env_para, algo_id, algo_para, new_executor=False, folder_num=1)
executor.test_model(test_timestep)

