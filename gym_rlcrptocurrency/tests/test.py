import gym
from gym_rlcrptocurrency.envs import Market
import numpy as np

# setup market data
data_path = "/Users/qzeng/Dropbox/MyDocument/Mac-ZQ/CS/CS234/Material2018/project/data/"
markets = [
    [Market("{:s}/bitstampUSD_1-min_data_2012-01-01_to_2018-01-08.csv".format(data_path))],
    [Market("{:s}/coinbaseUSD_1-min_data_2014-12-01_to_2018-01-08.csv".format(data_path))],
]

# setup environment
env = gym.make("rlcrptocurrency-v0")
env.set_markets(markets)

# initialize environment
init_portfolio = np.array(
    [
        [10000, 1],
        [10000, 1],
    ],
    dtype=np.float64
)
init_time = "2017-1-1"
env.init(init_portfolio, init_time)

# run environment
action_purchase = np.array(
    [
        [0.1],
        [-0.1],
    ],
    dtype=np.float64
)
action_transfer = np.array(
    [
        [[0], [0.1]],
        [[0], [0]],
    ]
)
# np.zeros(shape=(2, 2, 1), dtype=np.float64)
action = (action_purchase, action_transfer)

print "\nCompatible action?", env.check_obs_action(action)
obs, reward, done, _ = env.step(action)
print obs[0]
print obs[1]
print obs[2]
print reward
print done

for _ in range(60):
    # action_purchase = np.zeros(shape=(2, 1), dtype=np.float64)
    # action_transfer = np.zeros(shape=(2, 2, 1), dtype=np.float64)
    # action = (action_purchase, action_transfer)

    print "\nCompatible action?", env.check_obs_action(action)
    obs, reward, done, _ = env.step(action)
    print obs[0]
    print obs[1]
    print obs[2]
    print reward
    print done

