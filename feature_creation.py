import numpy as np
import pandas as pd

class features:
    def moving_average(self, window):
        ma = np.convolve(self.data['Close'], np.ones(window) / window, mode='valid')
        pad = np.full((window - 1,), np.nan)
        return np.concatenate((pad, ma))

    def __init__(self, stock_data):
        self.data = stock_data
        self.new_data = pd.DataFrame()
        for i in [14, 30, 50]:
            self.new_data['MovAv' + str(i)] = self.moving_average(i)

        self.new_data['Close'] = self.data['Close'].values

    def return_data(self):
        return(self.new_data)
