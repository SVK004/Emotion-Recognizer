import pickle
import librosa
import numpy as np

# with open('model_parameters.pkl', 'rb') as f:
#     W1, b1, W2, b2 = pickle.load(f)

# W1 = np.loadtxt("./w1.txt")
# b1 = np.loadtxt("./b1.txt")
# W2 = np.loadtxt("./w2.txt")
# b2 = np.loadtxt("./b2.txt")

# b1 = b1[:, :1]
# b2 = b2[:, :1]
# audio_file = "./03-01-05-02-02-01-20.wav"
from findAcc import test_prediction

def convert_and_make_prediction(audio_file, W1, b1, W2, b2):
    file, sr = librosa.load(audio_file, sr=None)

    
    print(file)
    # file = np.array(file)
    print(type(file))

    if(len(file > 176532)):
        file = file[:176532]

    padded_x = []
# for arr in x:
#     if arr.shape[0] > avg_size:
#         arr = arr[:avg_size]
#     # Ensure consistent padding calculation
#     if len(arr.shape) == 1:  # If array is 1D
    pad_width = (0, max(0, 176532 - len(file)))  # Pad to the right
    file = (np.pad(file, pad_width, mode='constant'))
    # else:
    #     raise ValueError("Input arrays are not uniformly 1D. Check dimensions!")
    # print("File shape:", file.shape)


    file = normalize(file)
    ans = test_prediction(file, W1, b1, W2, b2)

    return ans

def normalize(file):
    maxx = 0
    for i in range(len(file)):
        # for j in range(len(file[i])):
        # print("Length", len(file))
        maxx = max(maxx, abs(file[i]))
    
    for i in range(len(file)):
        file[i] = file[i] / maxx
    
    file = file.T
    return file


# def test_prediction(i, W1, b1, W2, b2):
#     print( f"Hello{i}")

def get_accuracy(W1, b1, W2, b2, audio_file):
    ans = convert_and_make_prediction(audio_file, W1, b1, W2, b2)
    ht = {0 : "neutral", 1 : "angry", 2 : "sad", 3 : "happy"}
    final_ans = []
    for i in ans:
        final_ans.append(ht.get(i))
    
    print(final_ans)
    return ht.get(ans[0])


# get_accuracy(W1, b1, W2, b2)
