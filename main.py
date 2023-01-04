import Gather_Data as gd
from LSTM_Engine import LSTM_Engine

ticker = "BTC"
start = "01-01-2022"
end = "01-01-2023"


period = "2y"
interval = "1d"

data = gd.GatherCryptoData(ticker, start, end)

LSTM_Engine(stock_data=data)




