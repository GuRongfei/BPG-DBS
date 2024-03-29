import time

import gym
import numpy as np
import tensorflow as tf

from stable_baselines import logger
from stable_baselines.common import explained_variance, tf_util
from BPG.base_class import ActorCriticRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.runners import AbstractEnvRunner
from BPG.policy import ActorCriticPolicy
from stable_baselines.common.schedules import get_schedule_fn
from stable_baselines.common.tf_util import total_episode_reward_logger
from stable_baselines.common.math_util import safe_mean


class BPG(ActorCriticRLModel):

    def __init__(self, policy, env, gamma=0.99, n_steps=128, ent_coef=0.01, learning_rate=2.5e-4, qv_coef=0.5,
                 lam=0.95, nminibatches=4, noptepochs=4,
                 verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None,
                 full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None):

        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.ent_coef = ent_coef
        self.qv_coef = qv_coef
        self.gamma = gamma
        self.lam = lam
        self.nminibatches = nminibatches
        self.noptepochs = noptepochs
        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log

        self.action_ph = None
        self.rewards_ph = None
        self.old_q_value_ph = None
        self.old_neglog_pac_ph = None
        self.grad_q_ph = None
        self.old_vpred_ph = None
        self.learning_rate_ph = None
        self.entropy = None
        self.qv_loss = None
        self.pg_loss = None
        self._train = None
        self.loss_names = None
        self.train_model = None
        self.act_model = None
        self.value = None
        self.n_batch = None
        self.summary = None


        super().__init__(policy=policy, env=env, verbose=verbose, requires_vec_env=True,
                         _init_setup_model=_init_setup_model, policy_kwargs=policy_kwargs,
                         seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)

        if _init_setup_model:
            self.setup_model()

    def _make_runner(self):
        return Runner(env=self.env, model=self, n_steps=self.n_steps,
                      gamma=self.gamma, lam=self.lam)

    def _get_pretrain_placeholders(self):
        policy = self.act_model
        if isinstance(self.action_space, gym.spaces.Discrete):
            return policy.obs_ph, self.action_ph, policy.policy
        return policy.obs_ph, self.action_ph, policy.deterministic_action

    def setup_model(self):
        with SetVerbosity(self.verbose):
            self.n_batch = self.n_envs * self.n_steps

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.set_random_seed(self.seed)
                self.sess = tf_util.make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)

                n_batch_step = None
                n_batch_train = None

                act_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1,
                                        n_batch_step, reuse=False, **self.policy_kwargs)
                with tf.variable_scope("train_model", reuse=True,
                                       custom_getter=tf_util.outer_scope_getter("train_model")):
                    train_model = self.policy(self.sess, self.observation_space, self.action_space,
                                              self.n_envs // self.nminibatches, self.n_steps, n_batch_train,
                                              reuse=True, **self.policy_kwargs)

                with tf.variable_scope("loss", reuse=False):
                    self.action_ph = train_model.pdtype.sample_placeholder([None], name="action_ph")
                    self.rewards_ph = tf.placeholder(tf.float32, [None], name="rewards_ph")
                    self.old_q_value_ph = tf.placeholder(tf.float32, [None], name="old_q_value_ph")
                    self.grad_q_ph = tf.placeholder(tf.float32, [None], name="grad_q_ph")
                    self.old_neglog_pac_ph = tf.placeholder(tf.float32, [None], name="old_neglog_pac_ph")
                    self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")

                    self.entropy = tf.reduce_mean(train_model.proba_distribution.entropy())

                    q_value = train_model.q_flat
                    qv_error = tf.square(q_value - self.rewards_ph)
                    self.qv_loss = .5 * tf.reduce_mean(qv_error)
                    self.new_q = q_value


                    mean = train_model.proba_distribution.mean
                    logstd = train_model.proba_distribution.logstd
                    distribution = tf.contrib.distributions.Normal(mean, tf.exp(logstd))
                    smaller_prob_sum = distribution.cdf(self.action_ph)
                    larger_prob_sum = 1 - smaller_prob_sum
                    sy_logprob_bpg = tf.log(larger_prob_sum * 1.0 / smaller_prob_sum)
                    self.pg_loss = -tf.reduce_mean(sy_logprob_bpg * self.grad_q_ph)
                    """the_prob = tf.exp(-neglogpac)
                    other_prob = tf.ones_like(the_prob) - the_prob
                    this_action = tf.cast(self.action_ph, dtype=tf.float32)
                    other_action = 1. - this_action
                    #smaller_prob = tf.math.multiply(this_action, other_prob) + tf.math.multiply(other_action, the_prob)
                    #larger_prob = 1 - smaller_prob
                    smaller_prob = other_action + tf.math.multiply(this_action, other_prob)
                    larger_prob = this_action + tf.math.multiply(other_action, other_prob)
                    sy_logprob_bpg = tf.log(larger_prob / smaller_prob)
                    self.pg_loss = -tf.reduce_mean(sy_logprob_bpg * self.grad_q_ph)"""

                    #loss = self.pg_loss - self.entropy * self.ent_coef + self.qv_loss * self.qv_coef
                    loss = self.qv_loss

                    self.mean = mean
                    self.std = train_model.proba_distribution.std

                    tf.summary.scalar('entropy_loss', self.entropy)
                    tf.summary.scalar('policy_gradient_loss', self.pg_loss)
                    tf.summary.scalar('q_value_loss', self.qv_loss)
                    tf.summary.scalar('loss', loss)

                    with tf.variable_scope('model'):
                        self.params = tf.trainable_variables()
                        if self.full_tensorboard_log:
                            for var in self.params:
                                tf.summary.histogram(var.name, var)
                    grads = tf.gradients(loss, self.params)
                    grads = list(zip(grads, self.params))
                trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph, epsilon=1e-5)
                self._train = trainer.apply_gradients(grads)

                self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy']

                with tf.variable_scope("input_info", reuse=False):
                    tf.summary.scalar('discounted_rewards', tf.reduce_mean(self.rewards_ph))
                    tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))

                    tf.summary.scalar('old_neglog_action_probability', tf.reduce_mean(self.old_neglog_pac_ph))
                    tf.summary.scalar('grad_q', tf.reduce_mean(self.grad_q_ph))
                    tf.summary.scalar('old_q_value', tf.reduce_mean(self.old_q_value_ph))

                    if self.full_tensorboard_log:
                        tf.summary.histogram('discounted_rewards', self.rewards_ph)
                        tf.summary.histogram('learning_rate', self.learning_rate_ph)
                        tf.summary.histogram('old_neglog_action_probability', self.old_neglog_pac_ph)
                        tf.summary.histogram('grad_q', self.grad_q_ph)
                        tf.summary.histogram('old_q_value', self.old_q_value_ph)
                        if tf_util.is_image(self.observation_space):
                            tf.summary.image('observation', train_model.obs_ph)
                        else:
                            tf.summary.histogram('observation', train_model.obs_ph)

                self.train_model = train_model
                self.act_model = act_model
                self.step = act_model.step
                self.proba_step = act_model.proba_step
                self.value = act_model.value
                self.initial_state = act_model.initial_state
                tf.global_variables_initializer().run(session=self.sess)  # pylint: disable=E1101

                self.summary = tf.summary.merge_all()

    def _train_step(self, learning_rate, obs, masks, actions, rewards, q_values, neglogpacs, grad_q, update,
                    writer, states=None):
        td_map = {self.train_model.obs_ph: obs,
                  self.action_ph: actions,
                  self.rewards_ph: rewards,
                  self.old_q_value_ph: q_values,
                  self.grad_q_ph: grad_q,
                  self.old_neglog_pac_ph: neglogpacs,
                  self.learning_rate_ph: learning_rate}
        if states is not None:
            td_map[self.train_model.states_ph] = states
            td_map[self.train_model.dones_ph] = masks

        if states is None:
            update_fac = max(self.n_batch // self.nminibatches // self.noptepochs, 1)
        else:
            update_fac = max(self.n_batch // self.nminibatches // self.noptepochs // self.n_steps, 1)

        if writer is not None:
            # run loss backprop with summary, but once every 10 runs save the metadata (memory, compute time, ...)
            if self.full_tensorboard_log and (1 + update) % 10 == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, policy_loss, value_loss, policy_entropy, _ = self.sess.run(
                    [self.summary, self.pg_loss, self.qv_loss, self.entropy, self._train],
                    td_map, options=run_options, run_metadata=run_metadata)
                writer.add_run_metadata(run_metadata, 'step%d' % (update * update_fac))
            else:
                summary, policy_loss, value_loss, policy_entropy, _ = self.sess.run(
                    [self.summary, self.pg_loss, self.qv_loss, self.entropy, self._train],
                    td_map)
            writer.add_summary(summary, (update * update_fac))
        else:
            policy_loss, value_loss, policy_entropy, _, new_q = self.sess.run(
                [self.pg_loss, self.qv_loss, self.entropy, self._train, self.new_q], td_map)

        print("-----------------------------------------------------")
        print("policy_loss: ", policy_loss)
        print("value_loss: ", value_loss)
        print("action: ", actions[:4])
        print("grad_q: ", grad_q[:4])
        print("q_value: ", q_values[:4])
        print("new_q: ", new_q[:4])
        print("reward: ", rewards[:4])

        return policy_loss, value_loss, policy_entropy

    def learn(self, total_timesteps, callback=None, log_interval=1, tb_log_name="PPO2",
              reset_num_timesteps=True):
        # Transform to callable if needed
        self.learning_rate = get_schedule_fn(self.learning_rate)

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        callback = self._init_callback(callback)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn()

            t_first_start = time.time()
            n_updates = total_timesteps // self.n_batch

            callback.on_training_start(locals(), globals())

            for update in range(1, n_updates + 1):
                assert self.n_batch % self.nminibatches == 0, ("The number of minibatches (`nminibatches`) "
                                                               "is not a factor of the total number of samples "
                                                               "collected per rollout (`n_batch`), "
                                                               "some samples won't be used."
                                                               )
                batch_size = self.n_batch // self.nminibatches
                t_start = time.time()
                frac = 1.0 - (update - 1.0) / n_updates
                lr_now = self.learning_rate(frac)

                callback.on_rollout_start()

                rollout = self.runner.run(callback)
                # Unpack
                obs, masks, actions, rewards, q_values, neglogpacs, grad_q, states, ep_infos = rollout

                callback.on_rollout_end()

                # Early stopping due to the callback
                if not self.runner.continue_training:
                    break

                self.ep_info_buf.extend(ep_infos)
                mb_loss_vals = []
                if states is None:  # nonrecurrent version
                    update_fac = max(self.n_batch // self.nminibatches // self.noptepochs, 1)
                    inds = np.arange(self.n_batch)
                    for epoch_num in range(self.noptepochs):
                        np.random.shuffle(inds)
                        for start in range(0, self.n_batch, batch_size):
                            timestep = self.num_timesteps // update_fac + ((epoch_num *
                                                                            self.n_batch + start) // batch_size)
                            end = start + batch_size
                            mbinds = inds[start:end]
                            slices = (arr[mbinds] for arr in (obs, masks, actions, rewards, q_values, neglogpacs, grad_q))
                            mb_loss_vals.append(self._train_step(lr_now, *slices, writer=writer,
                                                                 update=timestep))
                else:  # recurrent version
                    update_fac = max(self.n_batch // self.nminibatches // self.noptepochs // self.n_steps, 1)
                    assert self.n_envs % self.nminibatches == 0
                    env_indices = np.arange(self.n_envs)
                    flat_indices = np.arange(self.n_envs * self.n_steps).reshape(self.n_envs, self.n_steps)
                    envs_per_batch = batch_size // self.n_steps
                    for epoch_num in range(self.noptepochs):
                        np.random.shuffle(env_indices)
                        for start in range(0, self.n_envs, envs_per_batch):
                            timestep = self.num_timesteps // update_fac + ((epoch_num *
                                                                            self.n_envs + start) // envs_per_batch)
                            end = start + envs_per_batch
                            mb_env_inds = env_indices[start:end]
                            mb_flat_inds = flat_indices[mb_env_inds].ravel()
                            slices = (arr[mb_flat_inds] for arr in (obs, masks, actions, rewards, q_values, neglogpacs, grad_q))
                            mb_states = states[mb_env_inds]
                            mb_loss_vals.append(self._train_step(lr_now, *slices, update=timestep,
                                                                 writer=writer, states=mb_states))

                loss_vals = np.mean(mb_loss_vals, axis=0)
                t_now = time.time()
                fps = int(self.n_batch / (t_now - t_start))

                if writer is not None:
                    total_episode_reward_logger(self.episode_reward,
                                                rewards.reshape((self.n_envs, self.n_steps)),
                                                masks.reshape((self.n_envs, self.n_steps)),
                                                writer, self.num_timesteps)

                if self.verbose >= 1 and (update % log_interval == 0 or update == 1):
                    logger.logkv("serial_timesteps", update * self.n_steps)
                    logger.logkv("n_updates", update)
                    logger.logkv("total_timesteps", self.num_timesteps)
                    logger.logkv("fps", fps)
                    if len(self.ep_info_buf) > 0 and len(self.ep_info_buf[0]) > 0:
                        logger.logkv('ep_reward_mean', safe_mean([ep_info['r'] for ep_info in self.ep_info_buf]))
                        logger.logkv('ep_len_mean', safe_mean([ep_info['l'] for ep_info in self.ep_info_buf]))
                    logger.logkv('time_elapsed', t_start - t_first_start)
                    for (loss_val, loss_name) in zip(loss_vals, self.loss_names):
                        logger.logkv(loss_name, loss_val)
                    logger.dumpkvs()

            callback.on_training_end()
            return self

    def save(self, save_path, cloudpickle=False):
        data = {
            "gamma": self.gamma,
            "n_steps": self.n_steps,
            "qv_coef": self.qv_coef,
            "ent_coef": self.ent_coef,
            "learning_rate": self.learning_rate,
            "lam": self.lam,
            "nminibatches": self.nminibatches,
            "noptepochs": self.noptepochs,
            "verbose": self.verbose,
            "policy": self.policy,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "n_envs": self.n_envs,
            "n_cpu_tf_sess": self.n_cpu_tf_sess,
            "seed": self.seed,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs
        }

        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)


class Runner(AbstractEnvRunner):
    def __init__(self, *, env, model, n_steps, gamma, lam):
        super().__init__(env=env, model=model, n_steps=n_steps)
        self.lam = lam
        self.gamma = gamma

    def _run(self):
        # mb stands for minibatch
        mb_obs, mb_actions, mb_rewards, mb_q_values, mb_dones, mb_neglogpacs, mb_grad_q = [], [], [], [], [], [], []
        mb_states = self.states
        ep_infos = []
        for sample in range(self.n_steps):
            actions, q_values, self.states, neglogpacs, grad_q = self.model.step(self.obs, self.states, self.dones)
            """print("---------------------------------------------")
            print("action: ", actions)
            print("values: ", q_values)"""

            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_q_values.append(q_values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            mb_grad_q.append(grad_q)
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            #print("rewards: ", rewards)

            self.model.num_timesteps += self.n_envs

            if self.callback is not None:
                # Abort training early
                self.callback.update_locals(locals())
                if self.callback.on_step() is False:
                    self.continue_training = False
                    # Return dummy values
                    return [None] * 9

            for info in infos:
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    ep_infos.append(maybe_ep_info)
            mb_rewards.append(rewards)
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_actions = np.asarray(mb_actions)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_q_values = np.asarray(mb_q_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        mb_grad_q = np.asarray(mb_grad_q, dtype=np.float32)

        #mb_obs, mb_dones, mb_actions, mb_rewards, mb_q_values, mb_neglogpacs, mb_grad_q = \
            #map(swap_and_flatten, (mb_obs, mb_dones, mb_actions, mb_rewards, mb_q_values, mb_neglogpacs, mb_grad_q))

        return mb_obs, mb_dones, mb_actions, mb_rewards, mb_q_values, mb_neglogpacs, mb_grad_q, mb_states, ep_infos


# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def swap_and_flatten(arr):
    """
    swap and then flatten axes 0 and 1

    :param arr: (np.ndarray)
    :return: (np.ndarray)
    """
    shape = arr.shape
    return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])
