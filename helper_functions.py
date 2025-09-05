def ReLU(X):
  return np.maximum(0,X)

def ReLU_derivative(X):
  return (X>0).astype(float)

def softMax(Z):
  expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
  return expZ / np.sum(expZ, axis=1, keepdims=True)


def OneHotEncoding(Y):
  m = Y.shape[0]
  oneHotY = np.zeros((m , 10))
  oneHotY[np.arange(m), Y] = 1
  return oneHotY

def lossCalculation(Y, Y_hat):
    batchSize = Y.shape[0]
    eps = 1e-9  # avoid log(0)
    loss = -np.sum(Y * np.log(Y_hat + eps)) / batchSize
    return loss

def accuracy(X, Y, W1, B1, W2, B2, W3, B3):
    Z1 = X.dot(W1) + B1
    X2 = ReLU(Z1)
    Z2 = X2.dot(W2) + B2
    X3 = ReLU(Z2)
    Z3 = X3.dot(W3) + B3
    X4 = softMax(Z3)
    preds = np.argmax(X4, axis=1)
    return np.mean(preds == Y)

def forward_pass(X, W1, B1, W2, B2, W3, B3):
    Z1 = X.dot(W1) + B1
    X2 = ReLU(Z1)

    Z2 = X2.dot(W2) + B2
    X3 = ReLU(Z2)

    Z3 = X3.dot(W3) + B3
    X4 = softMax(Z3)

    return Z1, X2, Z2, X3, Z3, X4, X4

def backward_pass(X, Y, Z1, X2, Z2, X3, Z3, X4, W2, W3, m):
    # Output layer
    dZ3 = X4 - Y  # (m,10)
    dW3 = (1/m) * (X3.T.dot(dZ3))  # (64,10)
    dB3 = (1/m) * np.sum(dZ3, axis=0, keepdims=True)

    # Hidden layer 2
    dX3 = dZ3.dot(W3.T)  # (m,64)
    dRelUZ2 = ReLU_derivative(Z2)
    dZ2 = dX3 * dRelUZ2  # (m,64)
    dW2 = (1/m) * (X2.T.dot(dZ2))  # (128,64)
    dB2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)

    # Hidden layer 1
    dX2 = dZ2.dot(W2.T)  # (m,128)
    dRelUZ1 = ReLU_derivative(Z1)
    dZ1 = dX2 * dRelUZ1  # (m,128)
    dW1 = (1/m) * (X.T.dot(dZ1))  # (784,128)
    dB1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)

    return dW1, dB1, dW2, dB2, dW3, dB3


def update_parameters(W1, B1, W2, B2, W3, B3, dW1, dB1, dW2, dB2, dW3, dB3, alpha):
    W1 -= alpha*dW1
    B1 -= alpha*dB1
    W2 -= alpha*dW2
    B2 -= alpha*dB2
    W3 -= alpha*dW3
    B3 -= alpha*dB3
    return W1, B1, W2, B2, W3, B3

def predict(X):
    X2 = ReLU(X.dot(W1) + B1)
    X3 = ReLU(X2.dot(W2) + B2)
    X4 = softMax(X3.dot(W3) + B3)
    return np.argmax(X4, axis=1)


