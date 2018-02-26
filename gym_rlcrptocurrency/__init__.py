from gym.envs.registration import register

############################
# v0: baseline environment #
############################
# Only two exchanges
# Only BTC

register(
    id='rlcrptocurrency-v0',
    entry_point='gym_rlcrptocurrency.envs:RLCrptocurrencyEnv',
    kwargs=dict(n_exchange=2, n_currency=1, markets=None),
)