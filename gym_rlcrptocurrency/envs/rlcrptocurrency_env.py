import gym
from gym import error, spaces, utils
from gym.utils import seeding

import pandas as pd
import numpy as np


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
        ))
        self.action_space = spaces.Tuple((
            spaces.Box(low=-np.inf, high=np.inf, shape=(n_exchange, n_currency), dtype=np.float64),          # purchase
            spaces.Box(low=0.0, high=np.inf, shape=(n_exchange, n_exchange, n_currency), dtype=np.float64),  # transfer
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

    @property
    def n_exchange(self):
        return self._n_exchange

    @property
    def n_currency(self):
        return self._n_currency

    def set_markets(self, markets):
        self._state_market = markets
        return self

    def step(self, action):
        """
        Evolve the environment in one time-step

        :param action: A tuple of action_purchase and action_trasfer
            1. action_purchase is the buy/sell matrix (n_exchange, n_currency).
               Row represents exchanges, column represents crypto-currency.
               Value is the amount of purchase. Positive means buy-in; Negative means sell-off.
            2. action_trasfer is the transfer tensor of rank-3 (n_exchange, n_exchange, n_currency)
               First rank is the source exchanges;
               Second rank is the destination exchanges;
               Third rank is the crypto-currency
               Value is the amount of transfer, and must be non-negative

        :return: obs (obj), reward (float), episode_over(bool), info(dict)
        """

        #################
        # Handle Inputs #
        #################

        # basic validity check of input action
        assert self.action_space.contains(action), "Invalid input action!"

        # decompose actions
        action_purchase, action_transfer = action

        # "currency-conservation" check
        # To make sure we are market-free, we enforce conservation of crypto-currency across all exchanges
        # This is equivalent to requiring summation of a given crypto-currency purchase across exchanges to be zero
        # Notice that we only perform check here. It is still user's responsibility to make sure input action meet
        # such requirement (by, for example, reducing the degree of freedom)
        assert np.count_nonzero(np.sum(action_purchase, axis=0)) == 0, "Currency conservation is broken!"

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
        purchase_currency = action_purchase * market_price
        mask_buy = purchase_currency > 0
        mask_sell = purchase_currency < 0
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

                    self._state_portfolio[element.source, element.currency+1] -= element.amount
                    self._state_transfer.add(element)

                    # TODO: transfer fee is charged on cash account upon transfer
                    self._state_portfolio[element.source, 0] -= \
                        element.amount * market_price[exchange_source, currency] * self._fee_transfer

        ##########################
        # Check on updated state #
        ##########################

        assert self._check_market_align(), "Timestamp not aligned across markets!"

        ###################
        # Prepare returns #
        ###################

        _obs_ = self._get_observation()

        if not self.observation_space.contains(_obs_):
            _reward_ = -1e9  # TODO: infinity negative reward since this is not acceptable
            _done_ = True
        else:
            _reward_ = self._get_total_cash() - total_cash_before
            _done_ = False

        # TODO: nothing for now
        _info_ = None

        return _obs_, _reward_, _done_, _info_

    def init(self, init_portfolio, init_time):
        """
        Reset the environment and set starting point of the environment

        :param init_portfolio: Initial portfolio, same shape as self._state_portfolio
        :param init_time: Initial time, to initialize the market
        :return: Self
        """

        self._state_portfolio = init_portfolio
        self._state_transfer.reset()
        self._state_market = map(lambda market_exchange: map(lambda market: market.init(init_time), market_exchange),
                                 self._state_market)

        assert self._check_market_align(), "Invalid time initialization!"
        assert self.observation_space.contains(self._get_observation()), "Invalid portfolio initialization!"

        return self

    def reset(self):
        """
        A special init() with zero portfolio and default initial time (whatever first time-stamp in market data)

        :return: Self
        """

        return self.init(
            init_portfolio=np.zeros(shape=(self.n_exchange, self.n_currency+1), dtype=np.float64),
            init_time=None,
        )

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

        # TODO: no observation for buffer for now

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
        return portfolio, market_obs

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

    def _check_market_align(self):
        """
        Check if all markets is aligned at the same time

        :return: Boolean
        """

        current_times = \
            map(lambda market_exchange: set(map(lambda market: market.peek()["Timestamp"], market_exchange)),
                self._state_market)
        current_times = reduce(lambda x, y: x | y, current_times)

        return len(current_times) == 1

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
