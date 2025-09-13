#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recursive LSTM Prediction on Noisy Sine Wave
Created on Fri Sep 12 21:55:54 2025

@author: Sahar Jahani
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Generate the signal
n = 1500
t = np.linspace(1, 50, n)
true = np.sin(t)
noise = np.random.normal(0, 0.5, n)
noisy = true + noise

# Scale the signals to [0, 1]
Scaler = MinMaxScaler()
noisy_scaled = Scaler.fit_transform(noisy.reshape(-1, 1))
true_scaled = Scaler.transform(true.reshape(-1, 1))

# Create training windows
window_size = 70
def data_preparation(noisy, true, window_size):
    X, y = [], []
    for i in range(window_size, n):
        X.append(noisy[i - window_size:i])
        y.append(true[i])
    return np.array(X), np.array(y)

X, y = data_preparation(noisy_scaled, true_scaled, window_size)

# Split into training and test sets
split = int(0.7 * len(true))
trainX, trainy = X[:split], y[:split]
testX, testy = X[split:], y[split:]

# Define LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(window_size, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(trainX, trainy, epochs=40, batch_size=16, verbose=1)

# Recursive prediction loop
pred = []
inputs = testX[0, :]  # Initial input
prediction = 350  # Number of recursive steps

for _ in range(prediction):
    predict = model.predict(inputs.reshape((1, window_size, 1)), verbose=0)
    pred.append(predict)
    inputs = np.append(inputs[1:], predict)

# Inverse scale the prediction
pred = np.array(pred).reshape(-1, 1)
pred = Scaler.inverse_transform(pred).flatten()

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(range(len(noisy)), noisy, label='Noisy Signal', alpha=0.5)
plt.plot(range(len(true)), true, label='True Signal', linewidth=2, color='red')
plt.plot(range(window_size + split, window_size + split + prediction), pred, 
         label='Recursive LSTM Prediction', color='blue', linewidth=2)
plt.xlabel('Time Step', fontsize=14)
plt.ylabel('Amplitude', fontsize=14)
plt.title('Recursive LSTM Prediction of Noisy Sine Wave', fontsize=15, fontweight='bold')
plt.legend(fontsize=15, loc='upper right')
plt.ylim(-2.5, 3)
plt.grid(True)
plt.tight_layout()
plt.show()

