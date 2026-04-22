# 🧠 SwasthAI – AI Health Assistant

SwasthAI is a **multimodal AI-powered healthcare web application** that predicts diseases using **symptoms and medical images**, provides explanations, stores user records, and includes an AI chatbot for guidance.

---

## 🚀 Features

### 🩺 Disease Prediction

* Predict diseases based on selected symptoms
* Uses trained ML models (`model.pkl`, `svc.pkl`)
* Returns prediction with confidence score

### 📷 Medical Image Analysis

* Upload images (e.g., skin conditions)
* Uses deep learning model (`medical_image_model.h5`)
* Enhances prediction using image insights

### 🔍 Explainable AI

* Shows **important symptoms** contributing to prediction
* Helps users understand results

### 🤖 AI Chat Assistant

* Interactive chatbot for health queries
* Provides general precautions and suggestions

### 🗂 Records Management

* Stores predictions in `records.csv`
* View history via UI

---
## 📸 Project Screenshots & Insights

### 🧠 Disease Prediction Dashboard
<img width="1902" height="1037" alt="Image" src="https://github.com/user-attachments/assets/b90b159b-bf07-4097-aea6-d1e37c41e0ae" />
✔ Predicts disease using symptoms + AI  
✔ Shows confidence score  
✔ Provides explanation (why this disease)  
✔ Combines image + symptom intelligence  

---

### 🖼️ Medical Image Analysis
✔ Upload medical image (skin, etc.)  
✔ AI analyzes and predicts condition  
✔ Confidence score for image model  
✔ Fused with symptom prediction  

---

### 💬 AI Chat Assistant
(screenshots/chat.png)

✔ Ask health-related questions  
✔ Get precautions, diet, and guidance  
✔ Smart suggestions based on disease  

---

### 🏥 Treatment & Awareness Section
(screenshots/index.png)
✔ Step-by-step treatment guidance  
✔ Diet recommendations  
✔ Healthcare awareness (vaccines, etc.)  
✔ Preventive measures  

---
## 🛠️ Tech Stack

* **Frontend:** HTML, CSS
* **Backend:** Flask (Python)
* **ML Models:** Scikit-learn (Random Forest, SVC)
* **DL Models:** TensorFlow / Keras
* **Data Handling:** Pandas, NumPy

---

## 📁 Project Structure

```id="proj_struct"
AI_Nexus/
│
├── app.py                     # Main Flask application
│
├── models/                   # Saved ML & DL models
│   ├── model.pkl
│   ├── svc.pkl
│   ├── label_encoder.pkl
│   ├── medical_image_model.h5
│   └── medical_image_metadata.json
│
├── data/                     # Datasets
│   ├── (multiple datasets used for training)
│
├── templates/                # HTML pages
│   ├── home.html
│   ├── index.html
│   ├── chat.html
│   ├── records.html
│   ├── auth.html
│   ├── profile.html
│   └── terms.html
│
├── static/uploads/           # Uploaded images
│   ├── pic1.jpg
│   ├── pic2.jpeg
│
├── instance/
│   └── users.db              # Database for authentication
│
├── train_model.py            # Symptom model training
├── train_image_model.py      # Image model training
│
├── records.csv               # Prediction logs
├── .gitignore
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash id="clone"
git clone https://github.com/YOUR_USERNAME/SwasthAI.git
cd SwasthAI
```

### 2️⃣ Create Virtual Environment

```bash id="venv"
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3️⃣ Install Dependencies

```bash id="install"
pip install flask pandas numpy scikit-learn tensorflow
```

### 4️⃣ Run Application

```bash id="run"
python app.py
```

### 5️⃣ Open in Browser

```
http://127.0.0.1:5000
```

---

## 🧠 How It Works

1. User selects symptoms
2. (Optional) Uploads medical image
3. ML model predicts disease
4. Image model refines confidence
5. Explanation is generated
6. Data is stored in records

---

## ⚠️ Important Notes

* Image model predictions depend on training dataset
* Ensure models in `/models` folder are properly trained
* Dataset must be preprocessed before training

---

## 🚀 Future Improvements

* Use real medical datasets for higher accuracy
* Improve CNN model for skin disease detection
* Add NLP-based chatbot using LLM
* Deploy on cloud (Render / AWS)
* Mobile-friendly UI

---

## ⚠️ Disclaimer

This project is for **educational purposes only** and should not be used as a replacement for professional medical advice.
