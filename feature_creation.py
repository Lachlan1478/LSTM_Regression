import numpy as np
import pandas as pd

class features:
    def one_hot_encode(self, y_vals, criteria_value):
        new_y_vals = []
        for x in y_vals:
            if x > (criteria_value):
                new_y_vals.append(1)
            elif x < -(criteria_value):
                new_y_vals.append(-1)
            else:
                new_y_vals.append(0)
        return(new_y_vals)

    def RSI(self, window):
        # Calculate RSI with pandas library
        delta = self.data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean().abs()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return(rsi)

    def moving_average(self, window):
        ma = np.convolve(self.data['Target'], np.ones(window) / window, mode='valid')
        pad = np.full((window - 1,), np.nan)
        return np.concatenate((pad, ma))

    def __init__(self, stock_data, criteria):
        self.data = stock_data
        self.new_data = pd.DataFrame()

        self.data['5dFut'] = self.data['Close'].shift(-5)
        self.data['Target'] = self.data['5dFut'].pct_change(5)
        self.data['Volume'] = self.data['Volume'].pct_change(5)

        self.new_data = pd.DataFrame()
        for i in [14, 30, 50]:
            self.new_data['MovAv' + str(i)] = self.moving_average(i)

        self.new_data['Target'] = self.one_hot_encode(self.data['Target'].values, criteria)
        test = self.new_data['Target'].values
        vals = set(test)
        for val in vals:
            count = np.count_nonzero(test == val)
            print(f"{val}: {count}")
        self.new_data['Close'] = self.data['Close'].values
        self.new_data['RSI']  = self.RSI(14)
        self.new_data['Volume'] = self.data['Volume'].values

    def return_data(self):
        return(self.new_data)
