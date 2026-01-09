# Emotion Recognizer based on Voice Audio 🎙️🧠

This project is a **Speech Emotion Recognition (SER)** system that identifies human emotions from voice audio using machine learning and deep learning techniques. The system processes raw speech signals, extracts meaningful acoustic features, and classifies emotions using trained models.

The project demonstrates an **end-to-end ML pipeline** — from dataset preparation and preprocessing to feature extraction, model training, and evaluation.

---

## 📌 Project Objectives

- Analyze human speech signals to detect emotional states  
- Extract meaningful audio features such as MFCCs  
- Train machine learning / deep learning models for emotion classification  
- Evaluate model performance using accuracy and related metrics  
- Provide a clean and reproducible experimental setup  

---

## 🎭 Emotions Recognized

The system is designed to classify the following emotions (based on dataset availability):

- Neutral  
- Calm  
- Happy  
- Sad  
- Angry  
- Fearful  
- Disgust  
- Surprised  

---

## 📂 Dataset

- **Dataset Used:** RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)  
- **Source:** Kaggle  
- **Data Type:** Speech audio (`.wav` files)  
- **Speakers:** Multiple actors with labeled emotions  

> ⚠️ The dataset is **not included in this repository** due to size constraints.

---

## 🧠 Approach & Pipeline

1. **Dataset Ingestion**
   - Download and organize audio files
   - Traverse actor-wise directories
   - Identify emotion labels using filename encoding

2. **Preprocessing**
   - Audio loading and normalization
   - Basic noise handling
   - Emotion-wise segregation

3. **Feature Extraction**
   - MFCC (Mel-Frequency Cepstral Coefficients)
   - Additional spectral features (where applicable)

4. **Model Training**
   - CNN-based deep learning model
   - Experiments with different architectures and parameters

5. **Evaluation**
   - Accuracy calculation
   - Model comparison across experiments

---

## 🛠️ Tech Stack

### Programming Language
- Python

### Libraries & Frameworks
- NumPy  
- Pandas  
- Librosa  
- Scikit-learn  
- TensorFlow / Keras  
- Matplotlib  

### Development Tools
- Jupyter Notebook  
- VS Code  

---

## 📁 Repository Structure

```
Emotion-Recognizer/
├── Backend/ # Backend-related logic (if applicable)
├── Frontend/ # Frontend-related files (if applicable)
├── main.ipynb # Initial preprocessing & experiments
├── main_CNN.ipynb # CNN-based emotion recognition
├── main_redefined.ipynb # Refined experiments
├── main.py # Script-based execution
├── findAcc.py # Accuracy evaluation helper
├── README.md # Project documentation
└── .gitignore # Ignored files & folders


---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/SVK004/Emotion-Recognizer.git
cd Emotion-Recognizer

2️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt

3️⃣ Prepare Dataset
Download the RAVDESS dataset from Kaggle

Place the dataset in a local folder

Update dataset paths inside the code if required

4️⃣ Run the Model
bash
Copy code
python main.py
Or open the notebooks:

bash
Copy code
jupyter notebook
```
## 📊 Results
Multiple experiments were conducted using CNN-based architectures

Model accuracy varies based on feature set and parameters

Performance evaluation is available via findAcc.py

Detailed metrics such as confusion matrices and per-emotion accuracy can be added in future iterations.

## ⚠️ Notes
Trained model files and datasets are excluded from version control

This repository focuses on learning, experimentation, and pipeline design

Further optimization and deployment can be done in future versions

### 🎯 Future Improvements
```md
Real-time emotion recognition from microphone input

Improved noise robustness

Hyperparameter tuning

Deployment as a web or API-based service

Detailed evaluation visualizations

