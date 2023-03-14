import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns

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
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=optimization, metrics=['accuracy'])
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

        print("target value: ", self.y_train)
        print(self.X_train)

        # Create the grid search object
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, error_score='raise')
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
        # rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        # print("RMSE: ", rmse)

        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

        # evaluate the model on test data
        loss, accuracy = self.lstm_model.evaluate(self.X_test, self.y_test, verbose=0)

        # print the accuracy score
        print("Accuracy on test data: {:.2f}%".format(accuracy * 100))


    def plot_predictions(self, scaler, dates):
        # Make predictions on the test data
        y_pred = self.lstm_model.predict(self.X_test)

        # Invert the scaling to get the predictions back in the original scale
        #y_pred_inverted = scaler.inverse_transform(y_pred)

        # Get the actual values in the original scale
        #y_test_inverted = scaler.inverse_transform(self.y_test.reshape(-1, 1))

        self.actual_predictions = y_pred
        self.actual_results = self.y_test

        # Convert predictions from one-hot encoding to integer labels
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_test_labels = np.argmax(self.y_test, axis=1)

        # Generate the confusion matrix
        cm = confusion_matrix(y_test_labels, y_pred_labels)

        # Plot the confusion matrix as a heatmap
        class_names = ['Class 0', 'Class 1', 'Class 2']  # Replace with your own class labels
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.show()

        # n = 7  # Show one date label for every 7 data points
        # date_ticks = range(0, len(dates), n)
        #
        # Plot the predictions against the actual values
        # plt.plot(y_test_inverted, color='lightseagreen', label='Actual')
        # plt.plot(y_pred_inverted, color='red', label='Predicted')
        # plt.xticks(date_ticks, [dates[i] for i in date_ticks], rotation='vertical')
        #
        # # Add a title and labels to the x- and y-axis
        # plt.title('Actual vs. LSTM Predicted Values')
        # plt.xlabel('Date')
        # plt.ylabel('Value')
        #
        # plt.legend()
        # plt.show()


