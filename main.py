import Gather_Data as gd
import LSTM_Engine as lstm
import feature_creation as fc

ticker = "BTC"
start = "01-01-2022"
end = "01-01-2023"


period = "2y"
interval = "1d"

data = gd.GatherCryptoData(ticker, start, end)

#feature creation
feature_object = fc.features(data)
transformed_data = feature_object.return_data()

print(transformed_data)

#model
#model = lstm.lstm_model(stock_data=data)
#model.find_parameters()
#model.input_parameters(hidden_units=10, optimization='ADAM')
#model.model_to_use()
#model.plot_predictions()




