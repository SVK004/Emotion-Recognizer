#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


# !pip install librosa


# In[104]:


import numpy as np
import pandas as pd
import librosa


# In[105]:


# path = 'C:\\Users\\saive\\.cache\\kagglehub\\datasets\\uwrfkaggler\\ravdess-emotional-speech-audio\\versions\\1'
# path


# # In[106]:


# path


# # In[107]:


# import re
# import os
# import shutil

# destination = "D:\\Emotion_Recognizer_using_transformers\\neutral"
# subPaths = ["\\Actor_01", "\\Actor_02", "\\Actor_03", "\\Actor_04", "\\Actor_05", "\\Actor_06", "\\Actor_07", "\\Actor_08", "\\Actor_09", "\\Actor_10", "\\Actor_11", "\\Actor_12", "\\Actor_13", "\\Actor_14", "\\Actor_15", "\\Actor_16", "\\Actor_17", "\\Actor_18", "\\Actor_19", "\\Actor_20"]
# initPath = path

# for i in range(len(subPaths)):
#     path += subPaths[i]
#     for file in os.listdir(path):
#         filePath = os.path.join(path, file)
#         neutral = re.search(r'\d{2}-\d{2}-01-\d{2}-\d{2}-\d{2}-\d{2}', filePath)

#         if neutral and os.path.isfile(filePath):
#             destination_path = os.path.join(destination, file)
#             shutil.copyfile(filePath, destination_path)
#             print("Copied file:", file)
    
#     path = initPath

# path = initPath


# # In[108]:


# import re
# import os
# import shutil

# destination = "D:\\Emotion_Recognizer_using_transformers\\angry"
# subPaths = ["\\Actor_01", "\\Actor_02", "\\Actor_03", "\\Actor_04", "\\Actor_05", "\\Actor_06", "\\Actor_07", "\\Actor_08", "\\Actor_09", "\\Actor_10"]
# initPath = path

# for i in range(len(subPaths)):
#     path += subPaths[i]
#     for file in os.listdir(path):
#         filePath = os.path.join(path, file)
#         neutral = re.search(r'\d{2}-\d{2}-05-\d{2}-\d{2}-\d{2}-\d{2}', filePath)

#         if neutral and os.path.isfile(filePath):
#             shutil.copyfile(filePath, destination_path)
#             destination_path = os.path.join(destination, file)
#             print("Copied file:", file)
    
#     path = initPath


# # In[109]:


# import re
# import os
# import shutil

# destination = "D:\\Emotion_Recognizer_using_transformers\\sad"
# subPaths = ["\\Actor_01", "\\Actor_02", "\\Actor_03", "\\Actor_04", "\\Actor_05", "\\Actor_06", "\\Actor_07", "\\Actor_08", "\\Actor_09", "\\Actor_10"]
# initPath = path

# for i in range(len(subPaths)):
#     path += subPaths[i]
#     for file in os.listdir(path):
#         filePath = os.path.join(path, file)
#         neutral = re.search(r'\d{2}-\d{2}-04-\d{2}-\d{2}-\d{2}-\d{2}', filePath)

#         if neutral and os.path.isfile(filePath):
#             destination_path = os.path.join(destination, file)
#             shutil.copyfile(filePath, destination_path)
#             print("Copied file:", file)
    
#     path = initPath


# # In[110]:


# import re
# import os
# import shutil

# destination = "D:\\Emotion_Recognizer_using_transformers\\happy"
# subPaths = ["\\Actor_01", "\\Actor_02", "\\Actor_03", "\\Actor_04", "\\Actor_05", "\\Actor_06", "\\Actor_07", "\\Actor_08", "\\Actor_09", "\\Actor_10"]
# initPath = path

# for i in range(len(subPaths)):
#     path += subPaths[i]
#     for file in os.listdir(path):
#         filePath = os.path.join(path, file)
#         neutral = re.search(r'\d{2}-\d{2}-03-\d{2}-\d{2}-\d{2}-\d{2}', filePath)

#         if neutral and os.path.isfile(filePath):
#             shutil.copyfile(filePath, destination_path)
#             destination_path = os.path.join(destination, file)
#             print("Copied file:", file)
    
#     path = initPath


# In[111]:


neutral = []
sad = []
angry = []
happy=[]
y=[]


# In[112]:


import os
source = "D:\\Emotion_Recognizer_using_transformers\\neutral"

for file in os.listdir(source):
    # var, sr = librosa.load(source + "\\" + file)
    # neutral.append(var)
    neutral.append(librosa.load(source + "\\" + file, sr=None))
    y.append('neutral')


# In[113]:


source = "D:\\Emotion_Recognizer_using_transformers\\sad"

for file in os.listdir(source):
    # sad.append(librosa.tone(source + "\\" + file, sr=48000))
    # var, sr = librosa.load(source + "\\" + file, sr=22050, duration=2)
    # sad.append(librosa.tone(var, duration=2, sr=22050))
    
    # var, sr = librosa.load(source + "\\" + file)
    # sad.append(var)
    sad.append(librosa.load(source + "\\" + file, sr=None))
    y.append('sad')


# In[114]:


source = "D:\\Emotion_Recognizer_using_transformers\\angry"

for file in os.listdir(source):
    # angry.append(librosa.tone(source + "\\" + file, sr=48000))
    # var, sr = librosa.load(source + "\\" + file, sr=22050, duration=2)
    # angry.append(librosa.tone(var, duration=2, sr=22050))

    
    # var, sr = librosa.load(source + "\\" + file)
    # angry.append(var)
    angry.append(librosa.load(source + "\\" + file, sr=None))
    y.append('angry')


# In[115]:


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


# In[116]:


len(neutral)


# In[117]:


len(angry)


# In[118]:


len(sad)


# In[119]:


len(happy)


# In[120]:


x = []
for i in neutral:
    x.append(i[0])
for i in angry:
    x.append(i[0])
for i in sad:
    x.append(i[0])
for i in happy:
    x.append(i[0])


# In[121]:


len(x)


# In[122]:


y


# In[123]:


type(x[0])


# In[124]:


# x = x[0]
# for i in range(len(x)):
x[1]


# In[125]:


print(len(x))


# In[126]:


import sys
ans = sys.maxsize
for i in x:
    ans = min(len(i), ans)

ans


# In[127]:


for i in range(0, len(x)):
    x[i] = x[i][:ans]


# In[128]:


x


# In[129]:


for i in range(len(y)):
    if y[i] == 'neutral':
        y[i] = 0
    if y[i] == 'angry':
        y[i] = 1
    if y[i] == 'sad':
        y[i] = 2
    if y[i] == 'happy':
        y[i] = 3


# In[130]:


y


# In[131]:


y = np.array(y)


# In[132]:


y


# In[133]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# In[134]:


x_train = np.array(x_train)
x_test = np.array(x_test)


# In[135]:


type(x_train)


# In[136]:


type(y_train)


# In[137]:


type(y_train)


# In[138]:


# x_train = x_train.reshape(-1, 1)
# x_test = x_test.reshape(-1, 1)


# In[139]:


# from sklearn.ensemble import RandomForestClassifier
# rfc = RandomForestClassifier().fit(x_train, y_train)


# # In[140]:


# y_pred = rfc.predict(x_test)


# # In[141]:


# from sklearn.metrics import accuracy_score
# print(accuracy_score(y_test, y_pred))


# # In[143]:


# print(accuracy_score(y_train, rfc.predict(x_train)))


# # In[144]:


# from sklearn.svm import SVC

# svc = SVC().fit(x_train, y_train)


# # In[145]:


# new_y_pred = svc.predict(x_test)


# # In[146]:


# print(accuracy_score(y_test, new_y_pred))


# # In[147]:


# print(accuracy_score(y_train, svc.predict(x_train)))


# # In[148]:


# from sklearn.linear_model import LogisticRegression
# lr = LogisticRegression()

# lr.fit(x_train, y_train)


# # In[149]:


# y_pred_LR = lr.predict(x_test)
# accuracy_score(y_test, y_pred_LR)


# # In[150]:


# print(accuracy_score(y_train, lr.predict(x_train)))


# # In[151]:


# from sklearn.naive_bayes import GaussianNB
# gnb = GaussianNB()
# gnb.fit(x_train, y_train)


# # In[152]:


# gnb_y_pred = gnb.predict(x_test)


# # In[153]:


# accuracy_score(y_test, gnb_y_pred)


# # In[154]:


# print(accuracy_score(y_train, gnb.predict(x_train)))


# # In[48]:


# for i in x:
#     print(i, "\n\n")


# In[49]:


import numpy as np

new_x = np.vstack(x_train)


# In[50]:


new_x
m, n = new_x.shape


# In[51]:


new_x = new_x.T


# In[52]:


len(new_x[0])


# In[53]:


(new_x)


# In[54]:


max_val = -sys.maxsize - 1

for i in new_x:
    for j in i:
        max_val = max(max_val, abs(j))

max_val


# In[55]:


for i in range(len(new_x)):
    for j in range(len(new_x[i])):
        new_x[i][j] = new_x[i][j] / max_val


# In[56]:


y = y_train.T


# In[57]:


new_x


# In[58]:


# new_x = new_x[0]


# In[59]:


new_x.shape


# In[60]:


def init():
    W1 = np.random.rand(4, 147347) - 0.5
    b1 = np.random.rand(4, 255) - 0.5
    W2 = np.random.rand(4, 4) - 0.5
    b2 = np.random.rand(4, 255) - 0.5

    return W1, b1, W2, b2


# In[61]:


def ReLu(X):
    return np.maximum(X, 0)

def softmax(X):
    X_shifted = X - np.max(X, axis=0, keepdims=True)  # Shift for numerical stability
    exp_X = np.exp(X_shifted)
    return exp_X / np.sum(exp_X, axis=0, keepdims=True)


# In[62]:


def forward_propogation(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLu(Z1)

    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2


# In[63]:


def One_Hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T


# In[64]:


def deri_reLu(X):
    return X > 0


# In[65]:


def backward_propogation(Z1, Y, A1, A2, W2, X):
    one_hot_Y = One_Hot(Y)

    dZ2 = A2 - one_hot_Y
    dW2 = 1/ m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)

    dZ1 = W2.T.dot(dZ2) * deri_reLu(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)

    return db1, db2, dW1, dW2


# In[66]:


def update_params(db1, db2, dW1, dW2, W1, W2, b1, b2, alpha):
    W1 = W1 - dW1 * alpha
    W2 = W2 - dW2 * alpha
    
    b1 = b1 - db1 * alpha
    b2 = b2 - db2 * alpha

    return W1, W2, b1, b2


# In[67]:


def getPredictions(A2):
    return np.argmax(A2, 0)

def getAccuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size


# In[68]:


def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init()

    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_propogation(W1, b1, W2, b2, X)
        
        db1, db2, dW1, dW2 = backward_propogation(Z1, Y, A1, A2, W2, X)

        W1, W2, b1, b2 = update_params(db1, db2, dW1, dW2, W1, W2, b1, b2, alpha)

        if(i % 10 == 0):
            print(f"Iteration: {i}\t Accuracy = {getAccuracy(getPredictions(A2), Y)}")

    
    return W1, b1, W2, b2, A2


# In[ ]:


W1, b1, W2, b2, A2 = gradient_descent(new_x, y, 1000, 1)


# In[97]:


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


# In[94]:


new_x.shape


# In[95]:


x_test.shape


# In[84]:


new_x[0]


# In[85]:


maxx = 0
for i in range(len(x_test)):
    for j in range(len(x_test[i])):
        if(maxx < abs(x_test[i][j])):
            maxx = abs(x_test[i][j])


# In[86]:


max


# In[87]:


for i in range(len(x_test)):
    for j in range(len(x_test[i])):
        x_test[i][j] = x_test[i][j] / maxx


# In[90]:


x_test = x_test.T


# In[91]:


x_test.shape


# In[101]:


for i in range(64):
    test_prediction(i, W1, b1, W2, b2)


# In[103]:


test_accuracy = count / 64
print(test_accuracy)


# In[241]:
def get_accuracy(audioFile):
    return "Hello"

# %%
import pickle
with open('model_parameters.pkl', 'wb') as f:
    pickle.dump((W1, b1, W2, b2), f)
print("Model parameters saved to 'model_parameters.pkl'.")
# %%
W1
# %%
