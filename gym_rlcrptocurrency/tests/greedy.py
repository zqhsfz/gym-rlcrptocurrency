# Baseline greedy algorithm
# At each time-stamp, we pick the exchange-pair with largest positive reward
# If such exchange-pair does not exist, then we do nothing

import gym
from gym_rlcrptocurrency.envs import Market
import numpy as np
from copy import deepcopy
from tqdm import tqdm


class PolicyGreedy(object):
    def __init__(self, price_index, fee_exchange, fee_transfer):
        self._price_index = price_index
        self._fee_exchange = fee_exchange
        self._fee_transfer = fee_transfer

    def policy(self, obs):
        """
        Return an action given current observation

        :param obs: Observation as returned by environment
        :return: An action to take
        """

        obs_portfolio, obs_market, _ = obs
        n_currency = obs_portfolio.shape[1] - 1

        # generate action_list for each currency
        action_list = map(lambda currency: self._get_action_candidates(obs, currency), range(n_currency))
        action_list = reduce(lambda x, y: x + y, action_list)

        # get reward of each action, and select the final action
        action_reward_list = map(lambda action: (action, self._reward(action, obs)), action_list)
        action_reward_list = sorted(action_reward_list, key=lambda x: x[1], reverse=True)
        action_final, reward_final = action_reward_list[0]

        if reward_final > 0:
            return action_final
        else:
            return self._do_nothing(obs)

    def _reward(self, action, obs):
        """
        Return expected reward by taking action given the observation obs, assuming action is valid

        :param action: Action
        :param obs: Observation
        :return: Reward
        """

        action = deepcopy(action)

        _, obs_market, _ = obs
        action_purchase, _ = action

        mask_buy = action_purchase > 0
        mask_sell = action_purchase < 0

        market_price = obs_market[:, :, self._price_index]

        action_purchase[mask_buy] /= (1.0 - self._fee_transfer)
        purchase_currency = action_purchase * market_price
        purchase_currency[mask_buy] /= (1.0 - self._fee_exchange)
        purchase_currency[mask_sell] *= (1.0 - self._fee_exchange)

        reward = -np.sum(purchase_currency)
        return reward

    def _get_action_candidates(self, obs, currency):
        """
        Generate list of actions for a given currency

        :param obs: Observation
        :param currency: Currency
        :return: List of actions
        """

        obs = deepcopy(obs)

        obs_portfolio, obs_market, _ = obs
        n_exchange = obs_portfolio.shape[0]
        n_currency = obs_portfolio.shape[1] - 1

        action_list = []
        for exchange_from in range(n_exchange):
            for exchange_to in range(n_exchange):
                if exchange_from == exchange_to:
                    continue

                # purchase action
                # buy-in at "exchange_from", sell-off at "exchange_to"
                n_buy_max = obs_portfolio[exchange_from, 0] * (1. - self._fee_transfer) * (1. - self._fee_exchange) / obs_market[exchange_from, currency, self._price_index]
                n_sell_max = obs_portfolio[exchange_to, currency+1]
                n_final = min(n_buy_max, n_sell_max) * 0.99

                # lower clip on amount of crypto-currency purchase
                if n_final < 1e-3:
                    n_final = 0.

                # if price gap is too small, then we might not be able to exploit it in time.
                # In this case, we give up such window
                if np.isclose(
                        obs_market[exchange_from, currency, self._price_index],
                        obs_market[exchange_to, currency, self._price_index],
                        rtol=0.001,
                        atol=0.
                ):
                    n_final = 0.

                action_purchase = np.zeros(shape=(n_exchange, n_currency), dtype=np.float64)
                action_purchase[exchange_from, currency] = n_final
                action_purchase[exchange_to, currency] = -n_final

                # transfer action
                # transfer from "exchange_from" to "exchange_to"
                action_transfer = np.zeros(shape=(n_exchange, n_exchange, n_currency), dtype=np.float64)
                action_transfer[exchange_from, exchange_to, currency] = n_final

                # encapsulate action
                action = (action_purchase, action_transfer)
                action_list.append(action)

        return action_list

    @staticmethod
    def _do_nothing(obs):
        obs_portfolio, _, _ = obs
        n_exchange = obs_portfolio.shape[0]
        n_currency = obs_portfolio.shape[1] - 1

        action_purchase = np.zeros(shape=(n_exchange, n_currency), dtype=np.float64)
        action_transfer = np.zeros(shape=(n_exchange, n_exchange, n_currency), dtype=np.float64)

        return action_purchase, action_transfer


def run_policy():
    # setup market data
    data_path = "/Users/qzeng/Dropbox/MyDocument/Mac-ZQ/CS/CS234/Material2018/project/data/"
    markets = [
        [Market("{:s}/bitstampUSD_1-min_data_2012-01-01_to_2018-01-08.csv".format(data_path))],
        [Market("{:s}/coinbaseUSD_1-min_data_2014-12-01_to_2018-01-08.csv".format(data_path))],
    ]

    # setup environment
    env = gym.make("rlcrptocurrency-v1")
    env.set_markets(markets)

    # initialize environment
    init_portfolio = np.array(
        [
            [10000, 1],
            [10000, 1],
        ],
        dtype=np.float64
    )
    init_time = "2016-4-1"
    obs, reward, done, _ = env.init(init_portfolio, init_time)

    # setup agent
    agent = PolicyGreedy(env.market_obs_attributes.index("Weighted_Price"), env.fee_exchange, env.fee_transfer)

    # setup metrics
    reward_sum = reward

    # loop for a complete episode
    for _ in tqdm(range(100), desc="Loop on time-stamp"):
        action = agent.policy(obs)

        assert env.check_obs_action(action, verbose=True), "Invalid proposed action!"

        obs, reward, done, _ = env.step(action)
        reward_sum += reward

    # summary print out
    print "Initial balance:", env.init_balance
    print "Final balance:", env.get_balance()
    print "Reward accumulated:", reward_sum
    print "Return: {:.2f}%".format(100. * reward_sum / env.init_balance[0])


if __name__ == "__main__":
    run_policy()






