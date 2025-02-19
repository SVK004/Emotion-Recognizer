import numpy as np
import pandas as pd
import librosa

neutral = []
sad = []
angry = []
happy=[]
y=[]

import os
source = "D:\\Emotion_Recognizer_using_transformers\\neutral"

for file in os.listdir(source):
    # var, sr = librosa.load(source + "\\" + file)
    # neutral.append(var)
    neutral.append(librosa.load(source + "\\" + file, sr=None))
    y.append('neutral')


source = "D:\\Emotion_Recognizer_using_transformers\\sad"

for file in os.listdir(source):
    # sad.append(librosa.tone(source + "\\" + file, sr=48000))
    # var, sr = librosa.load(source + "\\" + file, sr=22050, duration=2)
    # sad.append(librosa.tone(var, duration=2, sr=22050))
    
    # var, sr = librosa.load(source + "\\" + file)
    # sad.append(var)
    sad.append(librosa.load(source + "\\" + file, sr=None))
    y.append('sad')


source = "D:\\Emotion_Recognizer_using_transformers\\angry"

for file in os.listdir(source):
    # angry.append(librosa.tone(source + "\\" + file, sr=48000))
    # var, sr = librosa.load(source + "\\" + file, sr=22050, duration=2)
    # angry.append(librosa.tone(var, duration=2, sr=22050))

    
    # var, sr = librosa.load(source + "\\" + file)
    # angry.append(var)
    angry.append(librosa.load(source + "\\" + file, sr=None))
    y.append('angry')

import os
source = "D:\\Emotion_Recognizer_using_transformers\\happy"

for file in os.listdir(source):
    # happy.append(librosa.tone(source + "\\" + file, sr=48000))
    # var, sr = librosa.load(source + "\\" + file, sr=22050, duration=2)
    # happy.append(librosa.tone(var, duration=2, sr=22050))

    
    # var, sr = librosa.load(source + "\\" + file, sr=None)
    # happy.append(var)
    happy.append(librosa.load(source + "\\" + file, sr=None))
    y.append('happy')


x = []
for i in neutral:
    x.append(i[0])
for i in angry:
    x.append(i[0])
for i in sad:
    x.append(i[0])
for i in happy:
    x.append(i[0])

import sys
ans = sys.maxsize
for i in x:
    ans = min(len(i), ans)


for i in range(0, len(x)):
    x[i] = x[i][:ans]

for i in range(len(y)):
    if y[i] == 'neutral':
        y[i] = 0
    if y[i] == 'angry':
        y[i] = 1
    if y[i] == 'sad':
        y[i] = 2
    if y[i] == 'happy':
        y[i] = 3


y = np.array(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

x_train = np.array(x_train)
x_test = np.array(x_test)

import numpy as np

new_x = np.vstack(x_train)

new_x
m, n = new_x.shape

new_x = new_x.T

max_val = -sys.maxsize - 1

for i in new_x:
    for j in i:
        max_val = max(max_val, abs(j))

for i in range(len(new_x)):
    for j in range(len(new_x[i])):
        new_x[i][j] = new_x[i][j] / max_val

y = y_train.T

def init():
    W1 = np.random.rand(4, 147347) - 0.5
    b1 = np.random.rand(4, 255) - 0.5
    W2 = np.random.rand(4, 4) - 0.5
    b2 = np.random.rand(4, 255) - 0.5

    return W1, b1, W2, b2


def ReLu(X):
    return np.maximum(X, 0)

def softmax(X):
    X_shifted = X - np.max(X, axis=0, keepdims=True)  # Shift for numerical stability
    exp_X = np.exp(X_shifted)
    return exp_X / np.sum(exp_X, axis=0, keepdims=True)


def forward_propogation(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLu(Z1)

    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2


def One_Hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T


def deri_reLu(X):
    return X > 0


def backward_propogation(Z1, Y, A1, A2, W2, X):
    one_hot_Y = One_Hot(Y)

    dZ2 = A2 - one_hot_Y
    dW2 = 1/ m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)

    dZ1 = W2.T.dot(dZ2) * deri_reLu(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)

    return db1, db2, dW1, dW2


def update_params(db1, db2, dW1, dW2, W1, W2, b1, b2, alpha):
    W1 = W1 - dW1 * alpha
    W2 = W2 - dW2 * alpha
    
    b1 = b1 - db1 * alpha
    b2 = b2 - db2 * alpha

    return W1, W2, b1, b2


def getPredictions(A2):
    return np.argmax(A2, 0)

def getAccuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size


def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init()

    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_propogation(W1, b1, W2, b2, X)
        
        db1, db2, dW1, dW2 = backward_propogation(Z1, Y, A1, A2, W2, X)

        W1, W2, b1, b2 = update_params(db1, db2, dW1, dW2, W1, W2, b1, b2, alpha)

        if(i % 10 == 0):
            print(f"Iteration: {i}\t Accuracy = {getAccuracy(getPredictions(A2), Y)}")

    
    return W1, b1, W2, b2, A2



W1, b1, W2, b2, A2 = gradient_descent(new_x, y, 1000, 1)


count = 0
def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_propogation(W1, b1, W2, b2, X)
    predictions = getPredictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    global count
    # current_image = x_test[:, index, None]
    prediction = make_predictions(x_test[:, index, None], W1, b1, W2, b2)
    label = y[index]
    ans = [0, 0, 0, 0]
    for i in prediction:
        ans[i]+=1
    print(ans)
    print("prediction: ", ans.index(max(ans)))
    print("Label: ", label)
    if(label == ans.index(max(ans))):
        count += 1


# import pickle
# with open('model_parameters.pkl', 'wb') as f:
#     pickle.dump((W1, b1, W2, b2), f)
# print("Model parameters saved to 'model_parameters.pkl'.")


def convert_and_make_prediction(audio_file, W1, b1, W2, b2):
    file = librosa.load(audio_file, sr=None)

    file = normalize(file)
    test_prediction(file, W1, b1, W2, b2)

def normalize(file):
    maxx = 0
    for i in range(len(file)):
        maxx = max(maxx, abs(file[i]))
    
    for i in range(len(x_test)):
            file[i] = file[i] / maxx
    
    file = file.T