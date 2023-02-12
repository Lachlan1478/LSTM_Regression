import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

class lstm_model:
    def __init__(self, stock_data):
        df = stock_data
        series = df['Close']

        # Drop any rows with missing data
        series = series.dropna()

        # Scale the data between 0 and 1
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.scaler.fit_transform(series.values.reshape(-1, 1))

        # Split the data into training and test sets
        training_data_len = int(len(scaled_data) * 0.8)
        train_data = scaled_data[0:training_data_len, :]
        test_data = scaled_data[training_data_len:, :]

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


    def find_parameters(self):

        # Create a function to build the LSTM model
        def build_model(hidden_units, optimization):
            model = Sequential()
            model.add(LSTM(units=hidden_units, return_sequences=True, input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
            model.add(LSTM(units=hidden_units, return_sequences=False))
            model.add(Dense(units=25))
            model.add(Dense(units=1))
            model.compile(loss='mean_squared_error', optimizer=optimization, metrics=['accuracy'])
            return model

        # Create the model wrapper
        model = KerasClassifier(build_fn=build_model, verbose=0)

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

        # Get the model's predicted prices
        predictions = grid_search.predict(self.X_test)
        predictions = self.scaler.inverse_transform(predictions)

        # Calculate the root mean squared error
        rmse = np.sqrt(np.mean(((predictions - self.y_test) ** 2)))

        # totalpctdiff = 0
        # for i in range(0, len(predictions) - 1):
        #     pctdiff = (predictions[i+1] - y_test) / y_test
        #     totalpctdiff = totalpctdiff + pctdiff


        print(rmse)