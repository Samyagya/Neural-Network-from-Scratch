import os
import pandas as pd
import numpy as np
from tensorflow.keras.datasets import mnist  #Only for importing data
from sklearn.model_selection import train_test_split
import helper_functions

X1, X_val, Y, y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=12)
OnehotY_val = helper_functions.OneHotEncoding(y_val)

m = X1.shape[0]
epochs = 1000
OnehotY = OneHotEncoding(Y)
alpha = 0.01
W1 = np.random.randn(784,128) * np.sqrt(2/784)
W2 = np.random.randn(128,64) * np.sqrt(2/128)
W3 = np.random.randn(64,10) * np.sqrt(2/64)
B1 = np.zeros((1,128))
B2 = np.zeros((1,64))
B3 = np.zeros((1,10))


for i in range(epochs):
    #Forward
    Z1, X2, Z2, X3, Z3, X4, _ = helper_functions.ANN_forward_pass(X1, W1, B1, W2, B2, W3, B3)

    #Loss
    loss = helper_functions.lossCalculation(OnehotY, X4)

    #Backward
    dW1, dB1, dW2, dB2, dW3, dB3 = helper_functions.ANN_backward_pass(X1, OnehotY, Z1, X2, Z2, X3, Z3, X4, W2, W3, m)

    #Update
    W1, B1, W2, B2, W3, B3 = helper_functions.ANN_update_parameters(W1, B1, W2, B2, W3, B3, dW1, dB1, dW2, dB2, dW3, dB3, alpha)

    if i % 10 == 0:
        train_acc = helper_functions.accuracy(X1, Y, W1, B1, W2, B2, W3, B3)
        val_acc = helper_functions.accuracy(X_val, y_val, W1, B1, W2, B2, W3, B3)
        print(f"Epoch {i}, Loss={loss:.4f}, Train Acc={train_acc:.4f}, Val Acc= {val_acc:.4f}")


y_pred = helper_functions.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print("Test accuracy:", accuracy)