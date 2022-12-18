from Gather_Data import GatherData
from LSTM_Engine import LSTM_Engine

stock = "WEB.AX"
period = "2y"
interval = "1d"

stock_data = GatherData(stock, period, interval)

LSTM_Engine(stock_data=stock_data)




