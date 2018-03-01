import gym
from gym import error, spaces, utils
from gym.utils import seeding

import pandas as pd
import numpy as np
from copy import deepcopy


class RLCrptocurrencyEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, n_exchange, n_currency, markets=None):
        """
        Crypto-currency arbitrage system environment

        :param n_exchange: Integer, representing number of exchanges
        :param n_currency: Integer, representing number of crypto-currencies to be considered. Notice that there is
                           always a flat currency (USD here) in addition to crypto-currency in portfolio as first column
        :param markets: A matrix of market with row being exchange, column being crypto-currency.
                        Market is an object as defined below
                        None by default. In which case, user should specify it through setter after instance is created
        """

        # define observation and action space
        # Since we have np.inf as bound, sample() on observation_space and action_space would throw error
        self.observation_space = spaces.Tuple((
            spaces.Box(low=0.0, high=np.inf, shape=(n_exchange, n_currency+1), dtype=np.float64),           # portfolio
            spaces.Box(low=0.0, high=np.inf,
                       shape=(n_exchange, n_currency, len(self.market_obs_attributes)), dtype=np.float64),  # market
            spaces.Box(low=0.0, high=np.inf, shape=(n_currency,), dtype=np.float64),                        # buffer
        ))
        self.action_space = spaces.Tuple((
            spaces.Box(low=-np.inf, high=np.inf, shape=(n_exchange, n_currency), dtype=np.float64),              # purchase
            spaces.Box(low=-np.inf, high=np.inf, shape=(n_exchange, n_exchange, n_currency), dtype=np.float64),  # transfer
        ))

        ##########
        # states #
        ##########

        # portfolio
        # row being exchange, and column being currency.
        # Value is the available balance on each account. Must be non-negative
        self._state_portfolio = np.zeros(shape=(n_exchange, n_currency+1), dtype=np.float64)

        # transfer buffer. See more details in class TransferBuffer
        self._state_transfer = TransferBuffer()

        # market status
        self._state_market = markets

        ####################
        # Some basic setup #
        ####################

        # TODO: just placeholder for now
        self._fee_exchange = 0.001   # 0.1%
        self._fee_transfer = 0.0005  # 0.05%

        #########################
        # Other internal states #
        #########################

        self._n_exchange = n_exchange
        self._n_currency = n_currency

        # Numpy array of shape (n_currency+1,), indicating the initial total balance
        # Should be reset every time init() is called
        self._init_balance = None

        # numerical tollerance
        self._tol = 1e-5

    @property
    def n_exchange(self):
        return self._n_exchange

    @property
    def n_currency(self):
        return self._n_currency

    @property
    def fee_exchange(self):
        return self._fee_exchange

    @property
    def fee_transfer(self):
        return self._fee_transfer

    @property
    def init_balance(self):
        return self._init_balance

    def set_markets(self, markets):
        self._state_market = markets
        return self

    def step(self, action):
        """
        Evolve the environment in one time-step

        :param action: A tuple of action_purchase and action_trasfer
            1. action_purchase is the buy/sell matrix (n_exchange, n_currency).
               Row represents exchanges, column represents crypto-currency.
               Value is the amount of crypto-currency being purchased. Positive means buy-in; Negative means sell-off.
            2. action_trasfer is the transfer tensor of rank-3 (n_exchange, n_exchange, n_currency)
               First rank is the source exchanges;
               Second rank is the destination exchanges;
               Third rank is the crypto-currency
               Value is the amount of crypto-currency being transferred, and must be non-negative

        :return: obs (obj), reward (float), episode_over(bool), info(dict)
        """

        #################
        # Handle Inputs #
        #################

        action = self._standardize_action(action)

        # action validity check
        assert self._check_action(action), "Invalid input action!"

        # decompose actions
        action_purchase, action_transfer = action

        #################
        # Update States #
        #################

        # get total cash in hand before the update
        total_cash_before = self._get_total_cash()

        # Extract current market information and then update market
        market_info = map(lambda market_exchange: map(lambda market: market.tick(), market_exchange),
                          self._state_market)
        market_price = np.array(
            map(lambda info_exchange: map(lambda info: info["Weighted_Price"], info_exchange), market_info),
            dtype=np.float64
        )

        # update transfer buffer
        transfer_finished = self._state_transfer.tick()

        # fill finished transfer to corresponding account
        for element in transfer_finished:
            self._state_portfolio[element.destination, element.currency+1] += element.amount

        # flat currency needed to spend on all buys / sells
        # There are two fee to be considered here:
        # 1. For an action of purchasing N crypto-currency, we actually need to buy N/(1-fee_transfer) crypto-currency,
        #    in order to account for the potential transfer loss in the future. For market-free arbitrage system, one
        #    transaction is expected. We enforce such constraint buy applying "currency-conservation" check.
        #    Notice that this does not apply for selling order though.
        # 2. For any buy/sell action, there is an exchange fee related to the amount of USD transaction.
        #    Notice the way we apply exchange fee on buy and sell is slightly different
        mask_buy = action_purchase > 0
        mask_sell = action_purchase < 0
        action_purchase[mask_buy] /= (1.0 - self._fee_transfer)

        purchase_currency = action_purchase * market_price
        purchase_currency[mask_buy] /= (1.0 - self._fee_exchange)
        purchase_currency[mask_sell] *= (1.0 - self._fee_exchange)

        # update flat currency balance
        self._state_portfolio[:, 0] -= np.sum(purchase_currency, axis=1)

        # update crypto-currency balance
        self._state_portfolio[:, 1:] += action_purchase

        # transfer crypto-currency
        for exchange_source in range(self.n_exchange):
            for exchange_destination in range(self.n_exchange):
                for currency in range(self.n_currency):
                    amount = action_transfer[exchange_source, exchange_destination, currency]
                    if amount <= 0.0:
                        continue

                    element = TransferElement(
                        source=exchange_source,
                        destination=exchange_destination,
                        amount=amount,
                        currency=currency,
                        time_left=1+int(np.random.exponential(scale=30)),  # TODO: placeholder only
                    )

                    # For an action of transfer N crypto-currency, we need to transfer slightly more
                    # in order to enforce N crypto-currency arriving at destination
                    self._state_portfolio[element.source, element.currency+1] -= (element.amount / (1.0 - self._fee_transfer))
                    self._state_transfer.add(element)

        ##########################
        # Check on updated state #
        ##########################

        assert self._check_state(), "Invalid state after one-step update!"

        ###################
        # Prepare returns #
        ###################

        _obs_ = self._get_observation()
        _reward_ = self._get_total_cash() - total_cash_before  # important: only cash
        _done_ = False  # infinite episode
        _info_ = None

        return _obs_, _reward_, _done_, _info_

    def init(self, init_portfolio, init_time):
        """
        Reset the environment and set starting point of the environment

        :param init_portfolio: Initial portfolio, same shape as self._state_portfolio
        :param init_time: Initial time, to initialize the market
        :return: Same output as step()
        """

        # deep copy of initialization so that same initialization can be repeated multiple times
        init_portfolio = deepcopy(init_portfolio)
        init_time = deepcopy(init_time)

        # initialize states
        self._state_portfolio = init_portfolio
        self._state_transfer.reset()
        self._state_market = map(lambda market_exchange: map(lambda market: market.init(init_time), market_exchange),
                                 self._state_market)

        # cache the initial total balance
        self._init_balance = np.sum(init_portfolio, axis=0)

        # validity checks
        assert self._check_state(), "Invalid state initialization!"

        # return
        return self._get_observation(), 0.0, False, None

    def reset(self):
        """
        A special init() with zero portfolio and default initial time (whatever first time-stamp in market data)

        :return: Self
        """

        raise NotImplementedError("Do not use reset()! Use init() instead.")

    def render(self, mode='human', close=False):
        pass

    ##########################

    def _get_total_cash(self):
        """
        Return the total amount of cash across all exchanges
        """

        return np.sum(self._state_portfolio[:, 0])

    def _get_observation(self):
        """
        Return observation based on current state

        :return: Tuple of (portfolio, market_obs)
          portfolio: A numpy array of shape (n_exchange, n_currency+1)
          market_obs: A numpy array of shape (n_exchange, n_currency, n_attributes)
        """

        # buffer
        # for buffer, we will only have aggregated balance for each crypto-currency
        # no time_left information should be used in order to reduce dependency on exact transfer mechanism
        buffer_obs = self._get_buffer_balance()

        # portfolio
        portfolio = self._state_portfolio

        # market observation
        def market_to_vector(market):
            market_info = market.peek()
            return map(lambda attribute: market_info[attribute], self.market_obs_attributes)

        market_obs = map(lambda exchange: map(lambda currency: market_to_vector(self._state_market[exchange][currency]),
                                              range(self.n_currency)),
                         range(self.n_exchange))
        market_obs = np.array(market_obs, dtype=np.float64)

        # return
        return portfolio, market_obs, buffer_obs

    @property
    def market_obs_attributes(self):
        return [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume_Crypto",
            "Volume_Flat",
            "Weighted_Price"
        ]

    def _get_buffer_balance(self):
        """
        Obtain the balance for all crypto-currency in transfer buffer

        :return: Numpy array of shape (n_currency,)
        """

        balance_transfer = np.zeros(shape=(self.n_currency,), dtype=np.float64)
        for element in self._state_transfer.buffer:
            balance_transfer[element.currency] += element.amount

        return balance_transfer

    def _standardize_action(self, action):
        """
        Standardize provided action

        :param action: Action
        :return: A copy of action, with transformation applied
        """

        # make a deep copy
        action = deepcopy(action)

        # decompose action
        action_purchase, action_transfer = action

        # remove all negative part in action_transfer
        action_transfer[action_transfer < 0.] = 0.

        # merge the redundant transfer (merge A->B and B->A into the direction of largest value)
        for currency in range(self.n_currency):
            for exchange_A in range(self.n_exchange):
                for exchange_B in range(self.n_exchange):
                    A_to_B = action_transfer[exchange_A, exchange_B, currency]
                    B_to_A = action_transfer[exchange_B, exchange_A, currency]

                    if A_to_B > B_to_A:
                        action_transfer[exchange_A, exchange_B, currency] = A_to_B - B_to_A
                        action_transfer[exchange_B, exchange_A, currency] = 0.
                    else:
                        action_transfer[exchange_B, exchange_A, currency] = B_to_A - A_to_B
                        action_transfer[exchange_A, exchange_B, currency] = 0.

        # return
        return action_purchase, action_transfer

    def _check_market_align(self):
        """
        Check if all markets are aligned at the same time

        :return: Boolean
        """

        current_times = \
            map(lambda market_exchange: set(map(lambda market: market.peek()["Timestamp"], market_exchange)),
                self._state_market)
        current_times = reduce(lambda x, y: x | y, current_times)

        return len(current_times) == 1

    def _check_crypto_balance(self):
        """
        Check if the total amount of crypto-currency across all exchanges (and transfer buffers) is no less than
        initial value

        :return: Boolean
        """

        balance_portfolio = np.sum(self._state_portfolio[:, 1:], axis=0)
        balance_transfer = self._get_buffer_balance()
        balance = balance_portfolio + balance_transfer

        return np.all(balance - self._init_balance[1:] >= -self._tol)

    def _check_state(self):
        """
        Combination of
        1. _check_market_align
        2. _check_crypto_balance
        3. observation space check

        :return: Boolean
        """

        return self._check_market_align() and self._check_crypto_balance() and self.observation_space.contains(self._get_observation())

    def _check_action(self, action):
        """
        Validity check of an action
        1. action space check
        2. Sum of crypto-currency orders across exchanges sum to zero

        :param action: Action
        :return: Boolean
        """

        return self.action_space.contains(action) and np.count_nonzero(np.sum(action[0], axis=0)) == 0

    def check_obs_action(self, action, obs=None, verbose=False):
        """
        Compatibility check between action and observation.
        The idea is that if one is able to pass this check, then running step() (immediately after)
        should not throw any exception

        :param obs: Observation. If None (default), will take the observation from current state
        :param action: Action.
        :param verbose: If true, then will print out failing reason
        :return: Boolean
        """

        action = self._standardize_action(action)

        # Some basic checks

        if not self._check_action(action):
            if verbose:
                print "\nFail on action validity check"
            return False

        if obs is None:
            obs = self._get_observation()

        if not self.observation_space.contains(obs):
            if verbose:
                print "\nFail on observation space basics check"
            return False

        # Compatibility check

        obs_portfolio, obs_market, obs_transfer = obs
        action_purchase, action_transfer = action

        # 1. There is enough crypto-currency in portfolio to sell
        mask_buy = action_purchase > 0
        mask_sell = action_purchase < 0

        if not np.all(obs_portfolio[:, 1:][mask_sell] + action_purchase[mask_sell] >= -self._tol):
            if verbose:
                print "\nNot enough crypto-currency to sell"
            return False

        # 2. There is enough cash in portfolio to buy
        price_index = self.market_obs_attributes.index("Weighted_Price")
        market_price = obs_market[:, :, price_index]

        action_purchase[mask_buy] /= (1.0 - self._fee_transfer)
        purchase_currency = action_purchase * market_price
        purchase_currency[mask_buy] /= (1.0 - self._fee_exchange)
        purchase_currency[mask_sell] *= (1.0 - self._fee_exchange)

        if not np.all(obs_portfolio[:, 0] - np.sum(purchase_currency, axis=1) >= -self._tol):
            if verbose:
                print "\nNot enough cash to buy crypto-currency"
            return False

        # 3. There should be enough crypto-currency left for transfer
        obs_portfolio_crypto_updated = obs_portfolio[:, 1:] + action_purchase
        action_transfer_from = np.sum(action_transfer, axis=1)
        action_transfer_from_adjusted = action_transfer_from / (1.0 - self._fee_transfer)
        obs_portfolio_crypto_updated -= action_transfer_from_adjusted

        if not np.all(obs_portfolio_crypto_updated >= -self._tol):
            if verbose:
                print "\nNot enough crypto-currency to transfer"
            return False

        # 4. Total crypto-currency balance larger than initialization after previous two steps
        balance_portfolio = np.sum(obs_portfolio_crypto_updated, axis=0)
        balance_transfer = obs_transfer + np.sum(action_transfer_from, axis=0)

        if not np.all(balance_portfolio + balance_transfer - self._init_balance[1:] >= -self._tol):
            if verbose:
                print "\nTotal crypto-currency balance less than initialization"
            return False

        return True

########################################################################################################################


class Market(object):
    def __init__(self, file_path):
        """
        Market object is responsible of providing:
          1. Market price
          2. Market observation

        Instead of a query of any time-stamp, we throw out in continuous manner for efficiency reason. User needs to
        set a starting point, and then object will throw out information step-by-step

        The only API for user is:
          1. init(): set up the initial point
          2. peek(): return current market information
          2. tick(): return current market information and move to next time-stamp


        :param file_path: Path to market data file.

        Loaded file will be converted to schematic data frame like object. In this class, we expect the input to be
        csv file with schema:

        Timestamp, Open, High, Low, Close, Volume_BTC, Volume_Currency, Weighted_Price

        Timestamp: Unix epoch time in unit of second.
                   Each row at a specific timestamp represents the aggregated data in (t-delta, t) interval
        Open, High, Low, Close: self-explained
        Volume_BTC: Transaction volume in cryptocurrency
        Volume_Currency: Transaction volume in flat currency
        Weighted_Price: The guide price for buy/sell at this time
        """

        print "Loading market data {:s} ...".format(file_path)

        # load and convert to pandas DataFrame
        data = pd.read_csv(file_path).copy()

        # unify columns
        data.columns = [
            "Timestamp",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume_Crypto",
            "Volume_Flat",
            "Weighted_Price"
        ]

        # sort by time
        data.sort_values(by="Timestamp", inplace=True)

        # convert timestamp to date_time object
        data["Timestamp"] = pd.to_datetime(data["Timestamp"], unit="s")

        # store data
        self._data = data

        # current time-stamp
        self._index = 0

    @property
    def data(self):
        return self._data

    def init(self, time):
        """
        Set initial time-stamp (t=0)

        :param time: Any time expression that is compatible with pd.to_datetime
        :return: self, for chaining
        """

        if time is None:
            self._index = 0
        else:
            # prepare inputs
            ts = self.data["Timestamp"].tolist()
            time = pd.to_datetime(time)

            # perform binary search
            i_start = 0
            i_end = len(ts)-1

            assert ts[i_start] <= time <= ts[i_end], "Input time out of bound!"

            i_match = None
            while True:
                i_current = (i_start + i_end)/2

                if (time == ts[i_current]) or (ts[i_current] < time < ts[i_current+1]):
                    i_match = i_current
                    break
                elif time < ts[i_current]:
                    i_end = i_current
                else:
                    i_start = i_current + 1

            assert i_match is not None, "Error in finding closest time-stamp!"

            self._index = i_match

        return self

    def peek(self):
        """
        Return current market information
        :return: A dict with key being attribute and value being corresponding value
        """

        info = self.data.iloc[self._index, :].to_dict()
        return info

    def tick(self):
        """
        Return current market information and move one-step forward

        :return: Market information at current time-step
        """

        info = self.peek()
        self._index += 1

        return info


class TransferElement(object):
    def __init__(self, source, destination, amount, currency, time_left):
        """
        Element in the transfer pipeline.
        """

        self._source = source
        self._destination = destination
        self._amount = amount
        self._currency = currency
        self._time_left = time_left

    @property
    def source(self):
        return self._source

    @property
    def destination(self):
        return self._destination

    @property
    def amount(self):
        return self._amount

    @property
    def currency(self):
        return self._currency

    @property
    def time_left(self):
        return self._time_left

    @property
    def done(self):
        """
        Whether the transfer for current buiffer is finished
        """
        return self._time_left == 0

    def tick(self):
        """
        Update self states for one time-stamp, and return self
        :return: Self
        """

        assert self._time_left > 0, "Do you forget to destroy current transfer buffer?"

        self._time_left -= 1
        return self


class TransferBuffer(object):
    def __init__(self):
        """
        buffer storing all pending transfer requests.
        This is essentially a list of TransferElement, sorted by time_left.

        The only API exposed to user is:
          1. add(): Add a new transfer element
          2. tick(): Evolve over the time, and return list of finished transfer elements
          3. reset(): Clear the buffer
        """

        self._buffer = []

    @property
    def size(self):
        return len(self._buffer)

    @property
    def buffer(self):
        return list(self._buffer)

    def add(self, element):
        """
        Add element into the pipeline

        :param element: A TransferElement object
        :return: Self
        """

        current_size = self.size

        if current_size == 0:
            self._buffer.append(element)
        else:
            if element.time_left < self._buffer[0].time_left:
                self._buffer.insert(0, element)
            elif element.time_left > self._buffer[current_size - 1].time_left:
                self._buffer.append(element)
            else:
                i_start = 0
                i_end = current_size - 1

                while True:
                    i_current = (i_start + i_end) / 2

                    if (self._buffer[i_current].time_left == element.time_left) or \
                            (self._buffer[i_current].time_left < element.time_left <= self._buffer[i_current+1].time_left):
                        self._buffer.insert(i_current+1, element)
                        break
                    elif element.time_left < self._buffer[i_current].time_left:
                        i_end = i_current - 1
                    else:
                        i_start = i_current + 2

        return self

    def tick(self):
        """
        Evolve the buffer by one time-step and return all transfer elements that are finished
        """

        # evolve each element
        self._buffer = map(lambda element: element.tick(), self._buffer)

        # remove all elements that have finished transfer
        output = []
        while self.size > 0 and self._buffer[0].done:
            output.append(self._buffer.pop(0))

        return output

    def reset(self):
        self._buffer = []
        return self
