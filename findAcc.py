import numpy as np

count = 0
def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_propogation(W1, b1, W2, b2, X)
    print("A2: ", A2)
    predictions = getPredictions(A2)
    print("Predictions:", predictions)
    return predictions

def test_prediction(file, W1, b1, W2, b2):
    # current_image = x_test[:, index, None]
    prediction = make_predictions(file, W1, b1, W2, b2)
    print("Predictionsss:", type(prediction))
    ans = [0, 0, 0, 0]
    for i in range(len(prediction)):
        ans[i] += prediction[i]
    # print(ans)
    max_val = max(ans)
    # finalAns = ans.index(max(ans))
    finalAns = [i for i, v in enumerate(ans) if v == max_val]
    print("prediction: ", finalAns)
    return finalAns


def forward_propogation(W1, b1, W2, b2, X):
    print("X_shape: ", X.shape)
    print("W1_shape: ", W1.shape)
    print("b1_shape: ", b1.shape)
    Z1 = W1.dot(X) + b1
    A1 = ReLu(Z1)

    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2


def ReLu(X):
    return np.maximum(X, 0)

def softmax(X):
    X_shifted = X - np.max(X, axis=0, keepdims=True)  # Shift for numerical stability
    exp_X = np.exp(X_shifted)
    return exp_X / np.sum(exp_X, axis=0, keepdims=True)

def getPredictions(A2):
    return np.argmax(A2, 1)