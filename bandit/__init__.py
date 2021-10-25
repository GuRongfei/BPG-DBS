from gym.envs.registration import register

register(
    id='continuous-armed_bandit-v0',
    entry_point='bandit.envs:ContBanditEnv',
)
