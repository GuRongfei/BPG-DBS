import time

import gym
import numpy as np
import tensorflow as tf

from stable_baselines import logger
from stable_baselines.common import explained_variance, tf_util
from BPG.grf_policy import Actor, Critic
from BPG.base_class import ActorCriticRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.runners import AbstractEnvRunner
from BPG.policy import ActorCriticPolicy
from stable_baselines.common.schedules import get_schedule_fn
from stable_baselines.common.tf_util import total_episode_reward_logger
from stable_baselines.common.math_util import safe_mean


class BPG:
    def __init__(self, actor_model, critic_model, env, n_steps=100):
        self.actor_model = actor_model
        self.critic_model = critic_model
        self.env = env
        self.n_steps = n_steps
        self.seed = None

        self.setup_model()
        self.runner = self._make_runner()

    def _make_runner(self):
        return Runner(env=self.env, model=self, n_steps=self.n_steps)

    def setup_model(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.env.seed(self.seed)
            self.env.action_space.seed(self.seed)
            self.sess = tf_util.make_session(graph=self.graph)

            actor = self.actor_model(self.sess, self.env)
            critic = self.critic_model(self.sess, self.env)

            with tf.variable_scope("loss", reuse=False):
                self.actor_obs_ph = actor.obs_ph
                self.critic_obs_ph = critic.obs_ph
                self.critic_action_ph = critic.action_ph
                self.rewards_ph = tf.placeholder(tf.float32, [None, 1], name="rewards_ph")
                self.q_values_ph = tf.placeholder(tf.float32, [None, 1], name="q_values_ph")
                self.grad_q_ph = tf.placeholder(tf.float32, [None, 1], name="grad_q_ph")

                self.entropy = actor.entropy()

                q_value = critic.q_value
                qv_error = tf.square(q_value - self.rewards_ph)
                self.qv_loss = .5 * tf.reduce_mean(qv_error)
                self.new_q = q_value

                mean = actor.mean
                logstd = actor.logstd
                distribution = tf.contrib.distributions.Normal(mean, tf.exp(logstd))
                smaller_prob_sum = distribution.cdf(self.critic_action_ph)
                larger_prob_sum = 1 - smaller_prob_sum
                sy_logprob_bpg = tf.log(larger_prob_sum * 1.0 / smaller_prob_sum)
                #sy_logprob_bpg = tf.clip_by_value(sy_logprob_bpg, -1, 1)
                self.pg_loss = -tf.reduce_mean(sy_logprob_bpg * self.grad_q_ph)

                self.logprob = sy_logprob_bpg
                self.logstd = logstd
                self.mean = mean

                loss = self.pg_loss + self.qv_loss * 0.5 - self.entropy * 0.001

                with tf.variable_scope('model'):
                    self.params = tf.trainable_variables()
                grads = tf.gradients(loss, self.params)
                grads = list(zip(grads, self.params))
            trainer = tf.train.AdamOptimizer(learning_rate=1e-2, epsilon=1e-5)
            self._train = trainer.apply_gradients(grads)

            self.step = actor.step
            self.predict_val = critic.predict_val
            tf.global_variables_initializer().run(session=self.sess)  # pylint: disable=E1101

            self.summary = tf.summary.merge_all()

    def _train_step(self, obs, actions, rewards, q_values, grad_q):
        td_map = {self.actor_obs_ph: obs,
                  self.critic_obs_ph: obs,
                  self.critic_action_ph: actions,
                  self.rewards_ph: rewards,
                  self.q_values_ph: q_values,
                  self.grad_q_ph: grad_q}

        policy_loss, value_loss, policy_entropy, log_prob, logstd, mean, _ = self.sess.run(
                [self.pg_loss, self.qv_loss, self.entropy, self.logprob, self.logstd, self.mean, self._train], td_map)

        print("-------------------------------------------")
        print("actions: ", actions)
        print("grad_q: ", grad_q)
        print("log_prob: ", log_prob)
        print("logstd: ", logstd)
        print("mean: ", mean)
        print("p_loss: ", policy_loss)
        print("v_loss: ", value_loss)
        print("entropy: ", policy_entropy)

        return policy_loss, value_loss, policy_entropy

    def learn(self, total_timesteps):
        n_updates = total_timesteps // self.n_steps

        for update in range(1, n_updates + 1):
            if not update%10:
                print("update: ", update, "/", n_updates)
            samples = self.runner.run()
            for epoch_num in range(4):
                self._train_step(*samples)

        return self


class Runner():
    def __init__(self, *, env, model, n_steps):
        self.env = env
        self.model = model
        self.n_steps = n_steps
        self.obs = env.reset()

    def run(self):
        mb_obs, mb_actions, mb_rewards, mb_q_values, mb_grad_q = [], [], [], [], []
        for sample in range(self.n_steps):
            actions = self.model.step(self.obs)
            q_values, grad_q = self.model.predict_val(self.obs, actions)

            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_q_values.append(q_values)
            mb_grad_q.append(grad_q)
            self.obs, rewards, _, _ = self.env.step(actions)
            mb_rewards.append(rewards[0])

        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_actions = np.asarray(mb_actions)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_q_values = np.asarray(mb_q_values, dtype=np.float32)
        mb_grad_q = np.asarray(mb_grad_q, dtype=np.float32)


        #mb_obs, mb_actions, mb_rewards, mb_q_values, mb_grad_q = \
        #    map(swap_and_flatten, (mb_obs, mb_actions, mb_rewards, mb_q_values, mb_grad_q))

        return mb_obs, mb_actions, mb_rewards, mb_q_values, mb_grad_q


def swap_and_flatten(arr):
    shape = arr.shape
    return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])
