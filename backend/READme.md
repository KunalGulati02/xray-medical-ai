# X-Ray Medical AI Backend

This backend provides the API endpoints for processing X-ray images and generating clinical-style summaries using a trained deep learning model.

## 📌 Features
- Flask-based REST API
- X-ray preprocessing pipeline
- Model inference for findings + simple explanation
- Supports PNG, JPG, JPEG, and DICOM (if enabled)
- Lightweight, modular code structure

## 🚀 How to Run

### 1. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate       # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Start the backend
```bash
python app.py
```

The backend will run at:
http://127.0.0.1:5000

## 📂 Folder Structure
backend/
│── app.py
│── model.py
│── train_model.py
│── data_split.py
│── requirements.txt

## 📮 API Endpoint
POST /analyze
Uploads an X-ray image and returns analysis.
