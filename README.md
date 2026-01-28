# Emotion Recognizer based on Voice Audio 🎙️🧠

A full-stack **Speech Emotion Recognition (SER)** system featuring a custom Neural Network implementation and a dual-backend architecture for robust data processing.

---

## 🚀 Key Highlights (The "Pro" Stuff)
- **Mathematical Implementation:** Built a Feed-Forward Neural Network from scratch using NumPy, featuring manual backpropagation and a numerically stable Softmax implementation.
- **Multi-Service Backend:** Utilizes a Node.js/Express gateway (`app.js`) for frontend communication and a Python/Flask service (`model_api.py`) for ML inference.
- **Hybrid Model Benchmarking:** Comparative research across SVM, Logistic Regression, Random Forest, and Naive Bayes to establish performance baselines.
- **Full-Stack Integration:** A React-based frontend dashboard (built with Bolt.ai) for seamless user interaction.

---

## 🧠 The Architecture & Pipeline

### 1. Feature Extraction
The system utilizes `librosa` to process raw `.wav` signals, extracting **MFCCs (Mel-Frequency Cepstral Coefficients)** and performing signal normalization.



### 2. Custom Neural Network Implementation
The core engine is a custom implementation designed for transparency:
- **Activation Functions:** ReLU for hidden layers and stable Softmax for the output layer.
- **Gradient Descent:** Manual backpropagation to update weights ($W_1, W_2$) and biases ($b_1, b_2$).

### 3. Dual-Backend Logic
- **Gateway (Node.js):** Handles frontend requests and manages the application state.
- **Inference Engine (Python):** Loads pre-trained parameters from `/models` and performs real-time emotion classification.

---

## 📂 Repository Structure

```text
Emotion-Recognizer/
├── project/                 # React Frontend (Bolt.ai)
│   ├── src/                 # UI Components, Pages, and Hooks
│   ├── public/              # Static assets (HTML, Favicon)
│   └── .gitignore           # Frontend-specific ignores (e.g., .bolt/)
├── Backend/                 # Dual-Backend Logic
│   ├── app.js               # Node.js Express Gateway (Frontend Entry Point)
│   ├── model_api.py         # Python Flask Service (Inference API)
│   ├── evaluate.py          # NN performance metrics (formerly findAcc.py)
│   └── package.json         # Node.js dependencies
├── notebooks/               # Research & Development
│   ├── 1_feature_extraction.ipynb
│   ├── 2_model_benchmarking.ipynb
│   └── 3_custom_nn_scratch.ipynb
├── models/                  # Trained Parameters & Checkpoints
│   ├── w1.txt               # Exported weight 1
│   ├── b1.txt               # Exported bias 1
│   └── model_parameters.pkl # Serialized full model
├── data/                    # Dataset Management (Local Only)
│   ├── raw/                 # Original, immutable audio files
│   └── processed/           # Normalized features ready for training
├── train.py                 # Benchmarking & Training Entry Point (formerly main.py)
├── config.py                # Centralized Path & Environment Settings
├── requirements.txt         # Python dependencies
├── .env                     # Local environment variables (dataset paths)
├── .gitignore               # Root-level ignore file
└── README.md                # Project documentation
```

## 🛠️ Installation & Setup
#### 1️⃣ Installation
```
git clone [https://github.com/SVK004/Emotion-Recognizer.git](https://github.com/SVK004/Emotion-Recognizer.git)
cd Emotion-Recognizer
pip install -r requirements.txt
```
#### 2️⃣ Configuration
Create a .env file in the root directory and set your local dataset path:
```
DATASET_PATH=D:\Your\Path\To\RAVDESS
```
#### 3️⃣ Running the API (Production Mode)
```
cd Backend
python app.py
```
The server will start at http://127.0.0.1:5000. Send a POST request to /process-audio with a .wav file to get an emotion prediction.

---

## 🎭 Emotions Recognized
The system classifies the following emotional states:
- Neutral 😐
- Angry 😡
- Sad 😔
- Happy 😊
---
## 📊 Results & Future Scope
- **Current Status:** The custom NN achieves competitive accuracy compared to standard ML models while offering full transparency into the decision-making process.

- **Next Step:** Implementing real-time emotion recognition via WebSockets for live microphone streams.

- **Scalability:** Moving from flat-file storage to a structured database for model versioning.