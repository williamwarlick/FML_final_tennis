import argparse
import pandas as pd
from TabularQLearner import TabularQLearner
import matplotlib.pyplot as plt










class StockEnvironment:

    def __init__(self):
        self.learner = TabularQLearner(states=128, actions=3,alpha=.001)

    def prepare_world (self, df):
        """
        Read the relevant price data and calculate some indicators.
        Return a DataFrame containing everything you need.
        """

        states = pd.DataFrame()
        states['S0'] = df['Prediction']
        states['S1'] = pd.qcut(df['Confidence'], q=2, labels=[0, 1]).astype(int) * 2
        states['S2'] = pd.qcut(df['Avg P1 Odds'], q=2, labels=[0, 1]).astype(int) * 4
        states['S3'] = pd.qcut(df['Avg P2 Odds'], q=2, labels=[0, 1]).astype(int) * 8
        states['S4'] = pd.qcut(df['P1 Rank'], q=2, labels=[0, 1]).astype(int) * 16
        states['S5'] = pd.qcut(df['P2 Rank'], q=2, labels=[0, 1]).astype(int) * 32
        states['S6'] = pd.qcut(df['Rank Diff'], q=2, labels=[0, 1]).astype(int) * 64


        return states

    def calc_state(self, df, match):
        """ Quantizes the state to a single number. """

        return df.loc[match].sum()


    def calc_reward(self, outcome, bet, odds):

        if bet:
            bet -= 1
            if outcome == bet:
                return odds * 1000 - 1000
            else:
                return -1000
        return 0


    def train_learner(self, data):
        states = self.prepare_world(data)

        for n in range(500):
            r = 0
            total_r = 0
            c = 0
            for i in range(1, 4000):
                state = self.calc_state(states, i)

                bet = self.learner.train(state, r)
                print(f"Bet is: {bet}")
                outcome = (data.loc[i, 'Outcome'])
                odds = data.loc[i, 'Avg P2 Odds']
                if outcome:
                    odds = data.loc[i, 'Avg P1 Odds']
                r = self.calc_reward(outcome, bet, odds)
                print(f"Reward is {r}")

                total_r += r
                c += 1
            print(f"avg trip reward {total_r / c} C:{c}")



    def test_learner(self, data):
        """
        Evaluate a trained Q-Learner on a particular stock trading task.
        Print a summary result of what happened during the test.

        Feel free to include portfolio stats or other information, but AT LEAST:

        Test trip, net result: $31710.00
        Benchmark result: $6690.0000
        """
        states = self.prepare_world(data)

        # Out of Sample Betting
        rs_out_of_sample = []
        x_out_of_sample = [i for i in range(len(data) - 3999)]
        current_value = 0
        for i in range(3999, len(data)):
            state = self.calc_state(states, i)
            bet = self.learner.test(state)
            outcome = data.loc[i, 'Outcome']
            odds = data.loc[i, 'Avg P2 Odds']
            if outcome:
                odds = data.loc[i, 'Avg P1 Odds']
            r = self.calc_reward(outcome, bet, odds)
            current_value += r
            rs_out_of_sample.append(current_value)

        plt.title('Out of Sample Betting')
        plt.ylabel('Strategy Value ($)')
        plt.xlabel("Number of Matches")
        plt.grid()
        plt.plot(x_out_of_sample, rs_out_of_sample)
        plt.show()

        # In Sample Betting
        rs_in_sample = []
        x_in_sample = [i for i in range(4000)]
        current_value = 0
        for i in range(4000):
            state = self.calc_state(states, i)
            bet = self.learner.test(state)
            outcome = data.loc[i, 'Outcome']
            odds = data.loc[i, 'Avg P2 Odds']
            if outcome:
                odds = data.loc[i, 'Avg P1 Odds']
            r = self.calc_reward(outcome, bet, odds)
            current_value += r
            rs_in_sample.append(current_value)
        print(len(x_in_sample), len(rs_in_sample))

        plt.title('In Sample Betting')
        plt.ylabel('Strategy Value ($)')
        plt.xlabel("Number of Matches")
        plt.grid()
        plt.plot(x_in_sample, rs_in_sample)
        plt.show()



data = pd.read_csv('with_odds.csv')

test_env = StockEnvironment()

test_env.train_learner(data)

test_env.test_learner(data)


"""
if __name__ == '__main__':
  # Load the requested stock for the requested dates, instantiate a Q-Learning agent,
  # and let it start trading.

  parser = argparse.ArgumentParser(description='Stock environment for Q-Learning.')

  date_args = parser.add_argument_group('date arguments')
  date_args.add_argument('--train_start', default='2018-01-01', metavar='DATE', help='Start of training period.')
  date_args.add_argument('--train_end', default='2019-12-31', metavar='DATE', help='End of training period.')
  date_args.add_argument('--test_start', default='2020-01-01', metavar='DATE', help='Start of testing period.')
  date_args.add_argument('--test_end', default='2021-12-31', metavar='DATE', help='End of testing period.')

  learn_args = parser.add_argument_group('learning arguments')
  learn_args.add_argument('--dyna', default=0, type=int, help='Dyna iterations per experience.')
  learn_args.add_argument('--eps', default=0.99, type=float, metavar='EPSILON', help='Starting epsilon for epsilon-greedy.')
  learn_args.add_argument('--eps_decay', default=0.99995, type=float, metavar='DECAY', help='Decay rate for epsilon-greedy.')

  sim_args = parser.add_argument_group('simulation arguments')
  sim_args.add_argument('--cash', default=200000, type=float, help='Starting cash for the agent.')
  sim_args.add_argument('--fixed', default=0.00, type=float, help='Fixed transaction cost.')
  sim_args.add_argument('--floating', default='0.00', type=float, help='Floating transaction cost.')
  sim_args.add_argument('--shares', default=1000, type=int, help='Number of shares to trade (also position limit).')
  sim_args.add_argument('--symbol', default='DIS', help='Stock symbol to trade.')
  sim_args.add_argument('--trips', default=500, type=int, help='Round trips through training data.')

  args = parser.parse_args()


  # Create an instance of the environment class.
  env = StockEnvironment( fixed = args.fixed, floating = args.floating, starting_cash = args.cash,
                          share_limit = args.shares )

  # Construct, train, and store a Q-learning trader.
  env.train_learner( start = args.train_start, end = args.train_end,
                     symbol = args.symbol, trips = args.trips, dyna = args.dyna,
                     eps = args.eps, eps_decay = args.eps_decay )

  # Test the learned policy and see how it does.

  # In sample.
  env.test_learner( start = args.train_start, end = args.train_end, symbol = args.symbol )

  # Out of sample.  Only do this once you are fully satisfied with the in sample performance!
  #env.test_learner( start = args.test_start, end = args.test_end, symbol = args.symbol )
"""""