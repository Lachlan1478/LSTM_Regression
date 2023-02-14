import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from datetime import datetime

class lstm_model:
    def __init__(self, stock_data):
        df = stock_data
        series = df['Close']


        dates = df['Date'].values
        dates = np.datetime_as_string(dates, unit='D', timezone='UTC')
        date_strings = [datetime.strptime(date, '%Y-%m-%d').strftime('%m/%d/%Y') for date in dates][::-1]

        # Drop any rows with missing data
        series = series.dropna()

        # Scale the data between 0 and 1
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.scaler.fit_transform(series.values.reshape(-1, 1))

        # Split the data into training and test sets
        training_data_len = int(len(scaled_data) * 0.8)
        train_data = scaled_data[0:training_data_len, :]
        test_data = scaled_data[training_data_len:, :]

        self.train_dates = date_strings[0:training_data_len]
        self.test_dates = date_strings[training_data_len:]

        # Create a function to create the training data for the LSTM model
        def create_training_data(data, look_back=1):
            X_train, y_train = [], []
            for i in range(len(data) - look_back - 1):
                a = data[i:(i + look_back), 0]
                X_train.append(a)
                y_train.append(data[i + look_back, 0])
            return np.array(X_train), np.array(y_train)

        # Set the number of time steps for the LSTM model
        look_back = 5
        self.X_train, self.y_train = create_training_data(train_data, look_back)
        self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], 1, self.X_train.shape[1]))

        # Create the testing data
        self.X_test, self.y_test = create_training_data(test_data, look_back)
        self.X_test = np.reshape(self.X_test, (self.X_test.shape[0], 1, self.X_test.shape[1]))

    def input_parameters(self, hidden_units, optimization):
        self.hidden_units = hidden_units
        self.optimization = optimization

    # Create a function to build the LSTM model
    def build_model(self, hidden_units, optimization):
        model = Sequential()
        model.add(LSTM(units=hidden_units, return_sequences=True, input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        model.add(LSTM(units=hidden_units, return_sequences=False))
        model.add(Dense(units=25))
        model.add(Dense(units=1))
        model.compile(loss='mean_squared_error', optimizer=optimization, metrics=['accuracy'])
        return model

    def find_parameters(self):
        # Create the model wrapper
        model = KerasClassifier(build_fn=self.build_model, verbose=0)

        print("Model build successfully!")

        # Define the hyperparameter grid
        hidden_units = [10, 50, 100]
        optimization = ['SGD', 'Adam', 'RMSprop']
        param_grid = dict(hidden_units=hidden_units, optimization=optimization)

        print("parameter grid built successfully!")

        # Create the grid search object
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
        # Fit the grid search to the data
        grid_search_result = grid_search.fit(self.X_train, self.y_train)

        print("Grid search Complete")

        # Print the best parameters and score
        print("Best parameters: {}".format(grid_search_result.best_params_))
        print("Best score: {:.2f}".format(grid_search_result.best_score_))

        self.hidden_units = grid_search_result.best_params_["hidden_units"]
        self.optimization = grid_search_result.best_params_["optimization"]


    def model_to_use(self):
        self.lstm_model = self.build_model(self.hidden_units, self.optimization)
        self.lstm_model.fit(self.X_train, self.y_train, epochs = 100,batch_size=10, verbose=0)

        # Get the model's predicted prices
        predictions = self.lstm_model.predict(self.X_test)
        predictions = self.scaler.inverse_transform(predictions)

        # Calculate the root mean squared error
        rmse = np.sqrt(np.mean(((predictions - self.y_test) ** 2)))

        print(rmse)

    def plot_predictions(self):
        # Make predictions on the test data
        y_pred = self.lstm_model.predict(self.X_test)

        # Invert the scaling to get the predictions back in the original scale
        y_pred_inverted = self.scaler.inverse_transform(y_pred)

        # Get the actual values in the original scale
        y_test_inverted = self.scaler.inverse_transform(self.y_test.reshape(-1, 1))

        n = 7  # Show one date label for every 7 data points
        date_ticks = range(0, len(self.test_dates), n)

        # Plot the predictions against the actual values
        plt.plot(y_test_inverted, color='lightseagreen', label='Actual')
        plt.plot(y_pred_inverted, color='red', label='Predicted')
        plt.xticks(date_ticks, [self.test_dates[i] for i in date_ticks], rotation='vertical')

        # Add a title and labels to the x- and y-axis
        plt.title('Actual vs. LSTM Predicted Values')
        plt.xlabel('Date')
        plt.ylabel('Value')

        plt.legend()
        plt.show()


