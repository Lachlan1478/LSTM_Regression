import numpy as np
import pandas as pd

class features:
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

    def __init__(self, stock_data, percentage_flag):
        self.data = stock_data
        self.new_data = pd.DataFrame()

        if(percentage_flag):
            self.data['5dFut'] = self.data['Close'].shift(-5)
            self.data['Target'] = self.data['5dFut'].pct_change(5)
            self.data['Volume'] = self.data['Volume'].pct_change(5)
        else:
            self.data['Target'] = self.data['Close'].shift(-5)

        self.new_data = pd.DataFrame()
        for i in [14, 30, 50]:
            self.new_data['MovAv' + str(i)] = self.moving_average(i)

        self.new_data['Target'] = self.data['Target'].values
        self.new_data['Close'] = self.data['Close'].values
        self.new_data['RSI']  = self.RSI(14)
        self.new_data['Volume'] = self.data['Volume'].values

    def return_data(self):
        return(self.new_data)
