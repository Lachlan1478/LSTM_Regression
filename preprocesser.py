import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler

class Preprocessor:
    def __init__(self, stock_data, features, target, look_back=5, train_test_split=0.8):
        self.stock_data = stock_data.dropna()
        self.look_back = look_back
        self.train_test_split = train_test_split
        self.features_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.train_data = None
        self.test_data = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.features = features
        self.target = target

    def create_training_data(self, data):
        X_train, y_train = [], []
        for i in range(len(data) - self.look_back - 1):
            a = data[i:(i + self.look_back), :-3]
            X_train.append(a)
            y_train.append(data[i + self.look_back, -3:])
        return np.array(X_train), np.array(y_train)

    def preprocess(self):
        df = self.stock_data

        features = df[self.features].values
        #target = df[self.target].values.reshape(-1, 1)

        target = to_categorical(df[self.target].values + 1, num_classes=3)
        #print(target)
        rows, cols = target.shape
        print("target length: ", rows, cols)

        self.features_scaler.fit(features)
        scaled_features = self.features_scaler.transform(features)
        #self.target_scaler.fit(target)
        #scaled_target = self.target_scaler.transform(target)

        #scaled_data = np.hstack((scaled_features, scaled_target))
        scaled_data = np.hstack((scaled_features, target))

        training_data_len = int(len(scaled_data) * self.train_test_split)
        self.train_data = scaled_data[0:training_data_len, :]
        self.test_data = scaled_data[training_data_len:, :]

        self.X_train, self.y_train = self.create_training_data(self.train_data)
        self.X_test, self.y_test = self.create_training_data(self.test_data)

        #print("y pre train data:", self.y_train)
        print("y pre train length: ", len(self.y_train))

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test