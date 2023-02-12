import Gather_Data as gd
import LSTM_Engine as lstm

ticker = "BTC"
start = "01-01-2022"
end = "01-01-2023"


period = "2y"
interval = "1d"

data = gd.GatherCryptoData(ticker, start, end)

model = lstm.lstm_model(stock_data=data)
model.find_parameters()



