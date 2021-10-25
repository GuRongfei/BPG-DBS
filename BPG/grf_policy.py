import warnings
from itertools import zip_longest
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from gym.spaces import Discrete

from stable_baselines.common.tf_util import batch_to_seq, seq_to_batch
from stable_baselines.common.tf_layers import conv, linear, conv_to_fc, lstm
from stable_baselines.common.distributions import make_proba_dist_type, CategoricalProbabilityDistribution, \
    MultiCategoricalProbabilityDistribution, DiagGaussianProbabilityDistribution, BernoulliProbabilityDistribution
from stable_baselines.common.input import observation_input


class BPGLearner:
    def __init__(self, sess, env):
        self.sess = sess
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space


class Actor(BPGLearner):
    def __init__(self, sess, env):
        super(Actor, self).__init__(sess, env)
        with tf.variable_scope("actor_input", reuse=False):
            self.obs_ph = tf.placeholder(shape=[None, self.observation_space.shape[0]], dtype=self.observation_space.dtype, name="actor_obs_ph")
            self.processed_obs = tf.cast(self.obs_ph, tf.float32)

        self.setup_init()

    def setup_init(self):
        with tf.variable_scope("model"):
            obs_latent = tf.tanh(linear(self.processed_obs, 'actor_1', 8))
            obs_latent = tf.tanh(linear(obs_latent, 'actor_2', 4))

            self.mean = linear(obs_latent, 'mean', self.action_space.shape[0])
            self.logstd = tf.get_variable(name='logstd', shape=[1, self.action_space.shape[0]])
            #pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)

        with tf.variable_scope("actor_output"):
            self.action = self.mean + tf.exp(self.logstd) * tf.random_normal(tf.shape(self.mean), dtype=self.mean.dtype)

    def step(self, obs):
        action, mean = self.sess.run([self.action, self.mean], {self.obs_ph: [obs]})
        print("mean: ", mean)
        return action[0]

    def entropy(self):
        entropy = tf.reduce_sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)
        return entropy


class Critic(BPGLearner):
    def __init__(self, sess, env):
        super(Critic, self).__init__(sess, env)
        with tf.variable_scope("critic_input", reuse=False):
            self.obs_ph = tf.placeholder(shape=[None, self.observation_space.shape[0]], dtype=self.observation_space.dtype, name="critic_obs_ph")
            self.processed_obs = tf.cast(self.obs_ph, tf.float32)
            self.action_ph = tf.placeholder(shape=[None, self.action_space.shape[0]], dtype=self.action_space.dtype, name="critic_action_ph")
            self.processed_action = tf.cast(self.action_ph, tf.float32)
            self.state_action = tf.concat([self.processed_obs, self.processed_action], axis=1)

        self.setup_init()

    def setup_init(self):
        with tf.variable_scope("model"):
            q_value = tf.tanh(linear(self.state_action, 'critic_1', 32))
            q_value = tf.tanh(linear(q_value, 'critic_2', 32))

        with tf.variable_scope("critic_output"):
            self.q_value = linear(q_value, 'critic_3', 1)
            self.grad_q = tf.gradients(self.q_value, self.processed_action)[0]

    def predict_val(self, obs, act):
        q_val_pred, grad_q = self.sess.run([self.q_value, self.grad_q], {self.obs_ph: [obs], self.action_ph: [act]})
        return q_val_pred[0], grad_q[0]

