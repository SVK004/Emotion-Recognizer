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