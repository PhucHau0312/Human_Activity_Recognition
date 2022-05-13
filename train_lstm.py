import numpy as np
import pandas as pd
import keras

from keras.layers import LSTM, Dense,Dropout
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

# Đọc dữ liệu
label = ["BODYSWING", "HANDSWING", "HEADSHAKING"]
no_of_label = len(label)

bodyswing_df = pd.read_csv(label[- no_of_label] + ".txt")
handswing_df = pd.read_csv(label[1 - no_of_label] + ".txt")
headshaking_df = pd.read_csv(label[2 - no_of_label] + ".txt")

X = []
y = []
no_of_timesteps = 50

dataset = bodyswing_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(0)

dataset = handswing_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(1)

dataset = headshaking_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(2)

X, y = np.array(X), np.array(to_categorical(y))
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model  = Sequential()
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units = int(no_of_label), activation = "sigmoid"))
model.compile(optimizer = "adam", metrics = "accuracy", loss = "categorical_crossentropy")

model.fit(X_train, y_train, epochs=30, batch_size=16,validation_data=(X_test, y_test))
model.save("model.h5")
