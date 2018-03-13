# Baseline greedy algorithm
# At each time-stamp, we pick the exchange-pair with largest positive reward
# If such exchange-pair does not exist, then we do nothing

import gym
from gym_rlcrptocurrency.envs import Market
import numpy as np
from copy import deepcopy
from tqdm import tqdm

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


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


def run_policy(env_name, start_date, n_days, n_test=1):
    """
    Run greedy algorithm for n_days starting from start_date

    :param env_name: Name of environment
    :param start_date: Str, starting date
    :param n_days: Int, number of days
    :param n_test: Int, numebr of tests to be repeated
    :return: array of (date, aggregated return rate)
    """

    # setup market data
    data_path = "/Users/qzeng/Dropbox/MyDocument/Mac-ZQ/CS/CS234/Material2018/project/data/"
    markets = [
        [Market("{:s}/bitstampUSD_1-min_data_2012-01-01_to_2018-01-08.csv".format(data_path))],
        [Market("{:s}/coinbaseUSD_1-min_data_2014-12-01_to_2018-01-08.csv".format(data_path))],
    ]

    # setup environment
    env = gym.make(env_name)
    env.set_markets(markets)

    # initialize environment
    init_portfolio = np.array(
        [
            [10000, 1],
            [10000, 1],
        ],
        dtype=np.float64
    )
    init_time = start_date
    env.init(init_portfolio, init_time)

    # setup agent
    agent = PolicyGreedy(env.market_obs_attributes.index("Weighted_Price"), env.fee_exchange, env.fee_transfer)

    # loop through number of tests
    reward_sum_list = []
    for _ in range(n_test):
        # reset env
        obs, reward, done, _ = env.init(init_portfolio, None)

        assert reward == 0., "How do you turn this on ..."

        # setup metrics
        reward_sum = reward
        output = []

        # loop for a complete episode
        for index_day in tqdm(range(n_days), desc="Loop on days"):
            for _ in range(1440):
                action = agent.policy(obs)
                assert env.check_obs_action(action, verbose=True), "Invalid proposed action!"
                obs, reward, done, _ = env.step(action)

                reward_sum += reward

            # reflect the accumulated return at the end of day
            output.append((index_day, 100.0 * reward_sum / 20000.))

        # update
        reward_sum_list.append(reward_sum)

    # summary print out
    print "Initial balance:", env.init_balance
    print "Reward accumulated: {:.4f} +/- {:.4f}".format(np.mean(reward_sum_list), np.std(reward_sum_list))
    # print "Return: {:.2f}%".format(100. * reward_sum / env.init_balance[0])

    return
    # return output


def sim_policy(env_name, start_date, episode, n_episode):
    """
    Run policy continuously to simulate the score during training
    """

    # setup market data
    data_path = "/Users/qzeng/Dropbox/MyDocument/Mac-ZQ/CS/CS234/Material2018/project/data/"
    markets = [
        [Market("{:s}/bitstampUSD_1-min_data_2012-01-01_to_2018-01-08.csv".format(data_path))],
        [Market("{:s}/coinbaseUSD_1-min_data_2014-12-01_to_2018-01-08.csv".format(data_path))],
        # [Market("{:s}/bitstampUSD_1-min_data_2012-01-01_to_2018-01-08.csv".format(data_path))],
    ]

    # setup environment
    env = gym.make(env_name)
    env.set_markets(markets)

    # initialize environment
    init_portfolio = np.array(
        [
            [10000, 1],
            [10000, 1],
        ],
        dtype=np.float64
    )
    init_time = start_date
    env.init(init_portfolio, init_time)

    # setup agent
    agent = PolicyGreedy(env.market_obs_attributes.index("Weighted_Price"), env.fee_exchange, env.fee_transfer)

    # setup metrics
    reward_episode_list = []
    t = 0
    for _ in tqdm(range(n_episode), desc="Loop on episodes"):
        env.init(init_portfolio, None)
        obs, reward, done, _ = env.move_market(t)
        reward_episode = 0.

        for _ in range(episode):
            action = agent.policy(obs)
            assert env.check_obs_action(action, verbose=True), "Invalid proposed action!"
            obs, reward, done, _ = env.step(action)
            reward_episode += reward

            t += 1

        reward_episode_list.append(reward_episode)

    print "Avg Reward: {:.4f} +/- {:.4f}".format(np.mean(reward_episode_list), np.std(reward_episode_list))


def plot_output(output_list, plot_path):
    """
    Visualize the output from run_policy()

    :param output_list: List of tuple (name, output), where output is the one as returned from run_policy()
    :param plot_path: Where to store the plot
    :return: No return
    """

    plt.figure()

    for name, output in output_list:
        x, y = zip(*output)
        plt.plot(x, y, label=name)

    plt.xlabel("Number of days")
    plt.ylabel("Accumulated return rate [%]")
    plt.legend()

    plt.savefig(plot_path)
    plt.close()


if __name__ == "__main__":
    # output1 = run_policy("rlcrptocurrency-v0", "2017-12-1", 30)
    # output1 = ("Mean transfer time 5 min", output1)
    #
    # output2 = run_policy("rlcrptocurrency-v1", "2017-12-1", 30)
    # output2 = ("Mean transfer time 30 min", output2)
    #
    # output3 = run_policy("rlcrptocurrency-v2", "2017-12-1", 30)
    # output3 = ("Mean transfer time 60 min", output3)
    #
    # output_list = [output1, output2, output3]
    # plot_output(output_list, "compare.png")

    # sim_policy("rlcrptocurrency-v0", "2015-3-1", 100, 10)
    # sim_policy("rlcrptocurrency-v1", "2015-3-1", 100, 10)
    # sim_policy("rlcrptocurrency-v1", "2017-12-5", 100, 10)
    # sim_policy("rlcrptocurrency-v0", "2015-8-23", 100, 100)
    # sim_policy("rlcrptocurrency-v0", "2017-1-1", 100, 10)
    
    # run_policy("rlcrptocurrency-v1", "2015-9-1", 7)
    # run_policy("rlcrptocurrency-v1", "2017-12-5", 7)
    run_policy("rlcrptocurrency-v1", "2017-11-15", 7, n_test=10)
    # run_policy("rlcrptocurrency-v1", "2017-11-5", 7)





