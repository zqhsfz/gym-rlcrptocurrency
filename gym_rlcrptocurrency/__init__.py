from gym.envs.registration import register

register(
    id='rlcrptocurrency-v0',
    entry_point='gym_rlcrptocurrency.envs:RLCrptocurrencyEnv',
)