# 🏥 Disease Prediction System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-black.svg?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

An intelligent, data-driven web application that leverages **Machine Learning** to predict potential diseases based on a user's symptoms and lifestyle habits. Powered by a high-accuracy **Random Forest Classifier** and wrapped in a premium, modern, responsive Flask web interface.

---

## ✨ Features

- 🧠 **97.6% Predictive Accuracy**: Highly-optimized Random Forest model trained on a comprehensive dataset of 132 distinct symptoms and 41 disease classes.
- 🥗 **Lifestyle Integration**: Augments predictions by dynamically factoring in personal habits (BMI, smoking status, alcohol consumption, exercise frequency, and water intake).
- ⚡ **Instantaneous Inference**: Processes symptom clusters and generates clinical predictions in under **50ms**.
- 🎨 **Premium Modern UI**: A fully responsive web interface featuring custom glassmorphism components, vibrant dark gradients, and fluid micro-animations.
- 📁 **Detailed Insights**: Detailed prediction results accompanied by structured medical disclaimers to ensure responsible user engagement.

---

## 🛠️ Tech Stack & Architecture

- **Backend Logic:** Flask (Python)
- **Machine Learning Core:** Scikit-Learn, NumPy, Pickle
- **Frontend Presentation:** HTML5, Modern Vanilla CSS3, JavaScript (interactive checklist & responsive styling)
- **Model Engine:** Random Forest Classifier (100 ensemble trees, balanced class weights)

---

## 📂 Repository Structure

```directory
├── app.py                     # Main Flask application & inference endpoint
├── dataset/
│   ├── Training.csv           # Cleaned training dataset (132 symptom columns)
│   └── Testing.csv            # Dedicated testing dataset for validation
├── model/
│   ├── disease_model.py       # ML Model training script
│   ├── disease_model.pkl      # Serialized Random Forest Classifier (~4.2MB)
│   └── columns.pkl            # Serialized list of symptom column headers
├── static/
│   ├── css/                   # Premium stylesheets with custom glassmorphism
│   └── images/                # Visual assets and graphics
└── templates/
    ├── home.html              # Modern, engaging landing page
    ├── index.html             # Symptom selector & lifestyle inputs form
    ├── about.html             # Project details and background
    ├── contact.html           # Professional outreach page
    └── result.html            # Diagnostic outcome & medical disclaimer page
```

---

## 🚀 Quick Start & Installation

### Prerequisites
- Python 3.8 or higher installed on your system.

### 1. Clone the Repository
```bash
git clone https://github.com/AnwarKhalid00/Disease_Prediction_System.git
cd Disease_Prediction_System
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install flask numpy scikit-learn
```

### 4. Run the Web Application
```bash
python app.py
```
Open your browser and navigate to **`http://127.0.0.1:5000`** to experience the application.

---

## ⚠️ Medical Disclaimer

> [!WARNING]
> **This application is a machine learning prototype designed for educational and informational purposes only.**
> It does not constitute medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider or medical professional for clinical concerns. Never disregard professional medical advice or delay seeking it because of something you read on this application.

---

## 📝 License

Distributed under the **MIT License**. See `LICENSE` for more information.
