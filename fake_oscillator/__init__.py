from gym.envs.registration import register

register(
    id='fake_oscillator-v0',
    entry_point='fake_oscillator.envs:FakeOscillatorEnv',
)
