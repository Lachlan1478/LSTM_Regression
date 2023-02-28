import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

class lstm_model:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    #Manual Function to input paramaeters for model to save time if find_parameters has already been run in past
    def input_parameters(self, hidden_units, optimization):
        self.hidden_units = hidden_units
        self.optimization = optimization

    # Create a function to build the LSTM model
    def build_model(self, hidden_units, optimization):
        model = Sequential()
        model.add(LSTM(units=hidden_units, return_sequences=True, input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        model.add(LSTM(units=hidden_units, return_sequences=False))
        model.add(Dense(units=hidden_units))
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


    def model_to_use(self, epochs_to_use = 100, batchsize_to_use = 10):
        self.lstm_model = self.build_model(self.hidden_units, self.optimization)
        history = self.lstm_model.fit(self.X_train, self.y_train, validation_split=0.2,
                                      epochs = epochs_to_use,batch_size=batchsize_to_use, verbose=0)

        # Make predictions on the test set
        y_pred = self.lstm_model.predict(self.X_test)

        # Calculate the RMSE
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        print("RMSE: ", rmse)

        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()


    def plot_predictions(self, scaler, dates):
        # Make predictions on the test data
        y_pred = self.lstm_model.predict(self.X_test)

        # Invert the scaling to get the predictions back in the original scale
        y_pred_inverted = scaler.inverse_transform(y_pred)

        # Get the actual values in the original scale
        y_test_inverted = scaler.inverse_transform(self.y_test.reshape(-1, 1))

        n = 7  # Show one date label for every 7 data points
        date_ticks = range(0, len(dates), n)

        # Plot the predictions against the actual values
        plt.plot(y_test_inverted, color='lightseagreen', label='Actual')
        plt.plot(y_pred_inverted, color='red', label='Predicted')
        plt.xticks(date_ticks, [dates[i] for i in date_ticks], rotation='vertical')

        # Add a title and labels to the x- and y-axis
        plt.title('Actual vs. LSTM Predicted Values')
        plt.xlabel('Date')
        plt.ylabel('Value')

        plt.legend()
        plt.show()


