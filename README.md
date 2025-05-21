# forest-fire-detection

# 🌲 Forest Fire Early Detection System

A simple machine learning-powered Streamlit app to detect forest fire risk using environmental inputs.

## Features
- Inputs: Temperature, Humidity, Wind, Rain
- Outputs: Fire Risk Prediction (Yes/No)
- Model: RandomForestClassifier

## How to Run Locally

```bash
pip install -r requirements.txt
python train_model.py  # Run once to train and save model
streamlit run app.py
