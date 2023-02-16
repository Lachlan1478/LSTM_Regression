import Gather_Data as gd
import LSTM_Engine as lstm
import feature_creation as fc
import preprocesser as pre
import numpy as np
from datetime import datetime

#----------------------------------#
#Setup
#----------------------------------#
ticker = "BTC"
start = "01-01-2020"
end = "01-01-2023"

data = gd.GatherCryptoData(ticker, start, end)
data = data.iloc[::-1]

#----------------------------------#
#feature creation
#----------------------------------#
feature_object = fc.features(data, False)
transformed_data = feature_object.return_data()

#----------------------------------#
#Preprocessing
#----------------------------------#
preprocessor = pre.Preprocessor(transformed_data,
                                features=['Target', 'RSI', 'Volume', 'MovAv14', 'MovAv30', 'MovAv50'],
                                target='Target',
                                look_back=5,
                                train_test_split=0.8)
preprocessor.preprocess()
X_train, y_train = preprocessor.get_train_data()
X_test, y_test = preprocessor.get_test_data()

print("The dimensions of the X_train are: ", X_train.shape)
print("The dimensions of the y_train are: ", y_train.shape)

#----------------------------------#
#Dates for Plotting
#----------------------------------#
dates = data['Date'].values
dates = np.datetime_as_string(dates, unit='D', timezone='UTC')
date_strings = [datetime.strptime(date, '%Y-%m-%d').strftime('%m/%d/%Y') for date in dates]

training_data_len = int(len(date_strings) * 0.8)
test_dates = date_strings[training_data_len:]
test_dates = test_dates[-len(y_test):]

#----------------------------------#
#model
#----------------------------------#
model = lstm.lstm_model(X_train, y_train, X_test, y_test)
model.find_parameters()

#model.input_parameters(hidden_units=50, optimization='Adam')

model.model_to_use(epochs_to_use=1000, batchsize_to_use= 20)
model.plot_predictions(scaler = preprocessor.target_scaler, dates = test_dates)




