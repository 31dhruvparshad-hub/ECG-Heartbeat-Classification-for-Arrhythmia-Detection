❤️ **ECG Heartbeat Classification with Personalized AI & Explainability**

📌 Overview

This project presents an deep learning pipeline for ECG heartbeat classification using the MIT-BIH Arrhythmia dataset. It combines a 1D Convolutional Neural Network (CNN) with patient-specific personalization and model explainability (Grad-CAM) to improve both performance and interpretability.

The system demonstrates how AI can move beyond generic predictions to personalized healthcare insights.
![66b3a1a27dbc3854c5257c8b_blog-heart-diagnostics](https://github.com/user-attachments/assets/aedcb896-28d7-4267-9eac-f2077991f8ac)


🚀 Key Features

📊 Automated ECG Data Pipeline

Signal preprocessing

Heartbeat segmentation

Patient-wise dataset construction

🧠 Deep Learning Model

Custom 1D CNN architecture for time-series ECG signals

Batch normalization, pooling, and dropout for stability

👤 Personalized Learning

Fine-tunes global model on patient-specific data

Improves classification for individual variability

🔍 Explainability (Grad-CAM / Saliency Maps)

Visualizes which parts of ECG signal influence predictions

Compares global vs personalized model focus

📈 Interactive Dashboard

Displays predictions and model explanations

Enables intuitive interpretation of results

🏗️ Project Structure
ecg-ai/
│── data/
│   ├── mitbih_loader.py
│   ├── heartbeat_segment.py
│   └── patient_data.py
│
│── models/
│   ├── cnn_model.py
│   ├── train_global.py
│   └── personalize.py
│
│── explainability/
│   └── gradcam.py
│
│── app/
│   └── dashboard.py
│
│── config/
│   └── settings.py
│
│── dataset/
│   └── processed/
│
│── main.py
│── requirements.txt
⚙️ Installation
git clone https://github.com/your-username/ecg-ai.git
cd ecg-ai
pip install -r requirements.txt
🧪 Dataset Preparation
python main.py

Processes MIT-BIH ECG records

Generates segmented heartbeat data

Stores .npy files for training

🧠 Model Training
python models/train_global.py

Trains the global CNN model

Saves trained weights (global_model.pth)

👤 Personalization
from models.personalize import personalize

model, train_data, test_data = personalize(record_id="233")

Fine-tunes the global model for a specific patient

Improves subject-specific performance

🔍 Explainability (Grad-CAM)
python explainability/gradcam.py

Generates saliency maps

Compares:

Global model attention

Personalized model attention

📊 Model Architecture

Input: 1D ECG signal

Conv Layers: 3 stacked Conv1D blocks

Activation: ReLU

Pooling: MaxPooling + AdaptiveAvgPooling

Fully Connected layers for classification

📈 Key Learnings

1D CNNs are highly effective for physiological time-series data

Personalization significantly improves prediction reliability

Explainability is critical in healthcare AI systems

Model focus varies between global and patient-specific training

🎯 Future Improvements

Real-time ECG streaming integration

Transformer-based sequence modeling

Multi-lead ECG support

Deployment as a web-based clinical tool

💡 Applications

Arrhythmia detection systems

AI-assisted cardiac diagnosis

Personalized healthcare analytics

Clinical decision support systems
