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
algo_id = 'ddpg'
train_timestep = int(2e4)
test_timestep = 2000
env_para = [{'len_state': 10}]
algo_para = [{'learning_rate': 2.5e-4, 'gamma': 0.99, 'ent_coef': 0.01, 'n_steps': 128, 'lam': 0.95}]

#executor = Executor(env_id, env_para, algo_id, algo_para)
#executor.train_policy(train_timestep)
#executor.test_model(test_timestep)


def mul_test():
    for i in range(len(env_para)):
        executor = Executor(env_id, env_para[i], algo_id, algo_para[i])
        executor.train_policy(train_timestep, save=True)
        executor.test_model(test_timestep)


if __name__=="__main__":
    mul_test()
