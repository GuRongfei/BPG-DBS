import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import gym
import gym_oscillator
import discrete_osc
from algo.BPG.grf_bpg import BPG
from algo.BPG.grf_policy import Actor, Critic

from algo.PPO.ppo2 import PPO2
from algo.PPO.ppo_policy import MlpPolicy
from algo.DQN.dqn import DQN
import algo.DQN.dqn_policy as dqn
from algo.DDPG.ddpg import DDPG
import algo.DDPG.ddpg_policy as ddpg
from stable_baselines.sac.sac import SAC
import stable_baselines.sac.policies as sac

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

class Executor:
    def __init__(self, env_id, env_para, algo_id, algo_para, new_executor=True, folder_num=-1, use_multi_env=False):
        self.env_id = env_id
        self.env_para = env_para
        self.algo_id = algo_id
        self.algo_para = algo_para
        self.new_executor = new_executor
        self.folder_num = folder_num
        self.use_multi_env = use_multi_env

        self.train_env = None
        self.test_env = None
        self.test_obs = None
        self.env_para_name = None
        self.env_para_value = None

        self.algo_model = None
        self.para_name = None
        self.para_value = None


        self.save_path = None

        self.train_timestep = None

        if self.new_executor:
            self.setup()
        else:
            self.load_setup()

    def setup(self):
        self.setup_env()
        self.algo_model = self.setup_algo_model()

        self.folder_num = sum([1 for _ in os.listdir('../result/executor/%s/%s/' % (self.env_id, self.algo_id))])
        self.save_path = '../result/executor/%s/%s/%s' % (self.env_id, self.algo_id, str(self.folder_num))
        os.mkdir(self.save_path)

    def setup_env(self):
        if self.env_id == 'oscillator-v0' or self.env_id == 'discrete-osc-v0':
            self.env_para_name = ['len_state']
            self.env_para_value = [250]
            for para_no in range(len(self.env_para_name)):
                if self.env_para_name[para_no] in self.env_para.keys():
                    self.env_para_value[para_no] = self.env_para[self.env_para_name[para_no]]
            self.train_env = gym.make(self.env_id, len_state=self.env_para_value[0])
            self.test_env = gym.make(self.env_id, len_state=self.env_para_value[0])
        else:
            self.train_env = gym.make(self.env_id)
            self.test_env = gym.make(self.env_id)

        self.test_obs = self.test_env.reset()

    def setup_algo_model(self):
        if self.algo_id == 'bpg':
            return BPG(Actor, Critic, self.train_env)
        elif self.algo_id == 'ppo':
            self.para_name = ['learning_rate', 'gamma', 'ent_coef', 'n_steps', 'lam']
            self.para_value = [2.5e-4, 0.99, 0.01, 128, 0.95]
            for para_no in range(len(self.para_name)):
                if self.para_name[para_no] in self.algo_para.keys():
                    self.para_value[para_no] = self.algo_para[self.para_name[para_no]]
            return PPO2(MlpPolicy, self.train_env,
                        learning_rate=self.para_value[0], gamma=self.para_value[1], ent_coef=self.para_value[2],
                        n_steps=self.para_value[3], lam=self.para_value[4])
        elif self.algo_id == 'dqn':
        elif self.algo_id == 'ddpg':
            return DDPG(ddpg.MlpPolicy, self.train_env)
        elif self.algo_id == 'sac':
            return SAC(sac.MlpPolicy, self.train_env)
        else:
            return None

    def load_setup(self):
        self.setup_env()
        self.algo_model = self.setup_algo_model()

        self.save_path = '../result/executor/%s/%s/%s' % (self.env_id, self.algo_id, str(self.folder_num))
        self.algo_model = self.algo_model.load('%s/model.pkl' % self.save_path)

    def train_policy(self, train_timestep, save=False):
        self.algo_model.learn(train_timestep)
        self.train_timestep = train_timestep

        fig = plt.figure(figsize=(25, 5))
        ax = fig.add_subplot(111)
        ax.plot(self.algo_model.episodes_reward, '-', label='State')
        ax.legend(loc=0)
        ax.grid()
        ax.set_xlabel("TimeStep(1024*)", fontsize=20)
        ax.set_ylabel("reward", fontsize=20)
        ax.set_title("Train Reward", fontsize=25)
        plt.savefig('%s/trainReward.png' % self.save_path)

        if save:
            self.algo_model.save('%s/model.pkl' % self.save_path)

    def test_model(self, test_timestep):
        if self.env_id == 'oscillator-v0':
            avr_rwd = self.test_osc(test_timestep)
        elif self.env_id == 'discrete-osc-v0':
            avr_rwd = self.test_dis(test_timestep)
        elif self.env_id =='continuous-armed_bandit-v0':
            avr_rwd = self.test_cab(test_timestep)
        else:
            test_obs, test_act, test_rwd = [], [], []
            for _ in range(test_timestep):
                action = self.model_pred()
                test_obs.append(self.test_obs)
                test_act.append(action)
                self.test_obs, rewards, dones, info = self.test_env.step(action)
                test_rwd.append(rewards)

            for timestep in range(test_timestep):
                print("------------------------------")
                print("|timestep    | %-15s|" % timestep)
                print("|observation | %-15s|" % test_obs[timestep])
                print("|action      | %-15s|" % test_act[timestep])
                print("|reward      | %-15s|" % test_rwd[timestep])

            avr_rwd = sum(test_rwd)/float(test_timestep)
            print("average reward: ", avr_rwd)
        if self.new_executor:
            self.savedata(round(avr_rwd, 3))

    def test_cab(self, test_timestep):
        test_obs, test_act, test_rwd = [], [], []
        for _ in range(test_timestep):
            action = self.model_pred()
            test_obs.append(self.test_obs)
            test_act.append(action[0])
            self.test_obs, rewards, dones, info = self.test_env.step(action)
            test_rwd.append(rewards)

            fig = plt.figure(figsize=(25, 10))
            ax = fig.add_subplot(111)
            ax.tick_params(labelsize=25)
            ax.plot(test_act, '-', c='lightcoral', label='Action')
            ax2 = ax.twinx()
            ax2.tick_params(labelsize=25)
            ax2.plot(test_obs, '-r', c='steelblue', label='State ')
            ax.legend(bbox_to_anchor=(1, 1), fontsize=25)
            ax.grid()
            ax.set_xlabel("TimeStep", fontsize=45)
            ax.set_ylabel("Actions", fontsize=45)
            ax2.set_ylabel("States", fontsize=45)
            ax2.legend(bbox_to_anchor=(1, 0.9), fontsize=25)
            ax.set_title("State length", fontsize=55)
            plt.savefig('%s/SA.png' % self.save_path)

        avr_rwd = sum(test_rwd) / float(test_timestep)
        return avr_rwd[0][0]

    def test_osc(self, test_timestep):
        test_obs, states_x, test_act, test_rwd = [], [], [], []
        for _ in range(1000):
            action = [0]
            test_obs.append(self.test_obs)
            states_x.append(self.test_env.x_val)
            test_act.append(action)
            self.test_obs, rewards, dones, info = self.test_env.step(action)
            test_rwd.append(rewards)

        for _ in range(test_timestep):
            action = self.model_pred()
            test_obs.append(self.test_obs)
            states_x.append(self.test_env.x_val)
            test_act.append(action)
            self.test_obs, rewards, dones, info = self.test_env.step(action)
            test_rwd.append(rewards)

        for _ in range(200):
            action = [0]
            test_obs.append(self.test_obs)
            states_x.append(self.test_env.x_val)
            test_act.append(action)
            self.test_obs, rewards, dones, info = self.test_env.step(action)
            test_rwd.append(rewards)

        print(states_x)

        fig = plt.figure(figsize=(25, 10))
        ax = fig.add_subplot(111)
        ax.tick_params(labelsize=25)
        ax.plot(test_act, '-', c='lightcoral', label='Action')
        ax2 = ax.twinx()
        ax2.tick_params(labelsize=25)
        ax2.plot(states_x, '-r', c='steelblue', label='State ')
        ax.legend(bbox_to_anchor=(1, 1), fontsize=25)
        ax.grid()
        ax.set_xlabel("TimeStep", fontsize=45)
        ax.set_ylabel("Actions", fontsize=45)
        ax2.set_ylabel("States", fontsize=45)
        ax2.legend(bbox_to_anchor=(1, 0.9), fontsize=25)
        ax.set_title("State length", fontsize=55)
        plt.savefig('%s/SA.png' % self.save_path)

        avr_rwd = sum(test_rwd[1000:-200]) / float(1000)
        return avr_rwd

    def test_dis(self, test_timestep):
        test_obs, states_x, test_act, test_rwd = [], [], [], []
        for _ in range(1000):
            action = 0
            test_obs.append(self.test_obs)
            states_x.append(self.test_env.x_val)
            test_act.append(action)
            self.test_obs, rewards, dones, info = self.test_env.step(action)
            test_rwd.append(rewards)

        for _ in range(test_timestep):
            action = self.model_pred()[0]
            test_obs.append(self.test_obs)
            states_x.append(self.test_env.x_val)
            self.test_obs, rewards, dones, info = self.test_env.step(action)
            test_act.append(self.test_env.action_val[action])
            test_rwd.append(rewards)

        for _ in range(200):
            action = 0
            test_obs.append(self.test_obs)
            states_x.append(self.test_env.x_val)
            test_act.append(action)
            self.test_obs, rewards, dones, info = self.test_env.step(action)
            test_rwd.append(rewards)

        fig = plt.figure(figsize=(25, 10))
        ax = fig.add_subplot(111)
        ax.tick_params(labelsize=25)
        ax.plot(test_act, '-', c='lightcoral', label='Action')
        ax2 = ax.twinx()
        ax2.tick_params(labelsize=25)
        ax2.plot(states_x, '-r', c='steelblue', label='State ')
        ax.legend(bbox_to_anchor=(1, 1), fontsize=25)
        ax.grid()
        ax.set_xlabel("TimeStep", fontsize=45)
        ax.set_ylabel("Actions", fontsize=45)
        ax2.set_ylabel("States", fontsize=45)
        ax2.legend(bbox_to_anchor=(1, 0.9), fontsize=25)
        ax.set_title("State length", fontsize=55)
        plt.savefig('%s/SA.png' % self.save_path)

        avr_rwd = sum(test_rwd[1000:-200]) / float(test_timestep)
        return avr_rwd


    def model_pred(self):
        predict = ['sac', 'ddpg']
        if self.algo_id in predict:
            action, _ = self.algo_model.predict(self.test_obs)
        elif self.algo_id == 'ppo':
            action, _, _, _ = self.algo_model.step([self.test_obs])
        else:
            action = self.algo_model.step(self.test_obs)
        return action

    def savedata(self, reward):
        print("saving para in %s" % self.save_path)
        with open("%s/param.txt" % self. save_path, 'w+') as f:
            f.write("--------------info----------------\n")
            f.write('|environment    | %-15s|\n' % self.env_id)
            f.write('|algorithm      | %-15s|\n' % self.algo_id)
            f.write("-------------env para-------------\n")
            for i in range(len(self.env_para_value)):
                f.write("|%-15s| %-15s|\n" % (self.env_para_name[i], str(self.env_para_value[i])))
            f.write("------------algo para-------------\n")
            f.write("|train timesteps| %-15s|\n" % self.algo_model.actual_trainstep)
            for i in range(len(self.para_value)):
                f.write("|%-15s| %-15s|\n" % (self.para_name[i], str(self.para_value[i])))
            f.write("--------------rslt----------------\n")
            f.write("|reward         | %-15s|" % reward)
