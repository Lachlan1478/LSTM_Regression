import numpy as np
import pandas as pd

class features:
    def moving_average(self, window):
        ma = np.convolve(self.data['Target'], np.ones(window) / window, mode='valid')
        pad = np.full((window - 1,), np.nan)
        return np.concatenate((pad, ma))

    def __init__(self, stock_data, percentage_flag):
        self.data = stock_data

        if(percentage_flag):
            self.data['5dFut'] = self.data['Close'].shift(-5)
            self.data['Target'] = self.data['5dFut'].pct_change(5)
        else:
            self.data['Target'] = self.data['Close']

        self.new_data = pd.DataFrame()
        for i in [14, 30, 50]:
            self.new_data['MovAv' + str(i)] = self.moving_average(i)

        self.new_data['Target'] = self.data['Target'].values
        self.new_data['Close'] = self.data['Close']

    def return_data(self):
        return(self.new_data)
