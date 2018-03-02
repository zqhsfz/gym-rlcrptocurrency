from gym.envs.registration import register

############################
# v0: baseline environment #
############################
# Only two exchanges
# Only BTC
# 5 minute mean transfer time

register(
    id='rlcrptocurrency-v0',
    entry_point='gym_rlcrptocurrency.envs:RLCrptocurrencyEnv',
    kwargs=dict(n_exchange=2, n_currency=1, mean_transfer_time=5, markets=None),
)

############################
# v1: baseline environment #
############################
# Only two exchanges
# Only BTC
# 30 minute mean transfer time

register(
    id='rlcrptocurrency-v1',
    entry_point='gym_rlcrptocurrency.envs:RLCrptocurrencyEnv',
    kwargs=dict(n_exchange=2, n_currency=1, mean_transfer_time=30, markets=None),
)