import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

st.title("🔥 Forest Fire Detection - Initial Model Trainer")

# Download and load data
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv"
    return pd.read_csv(url)

df = load_data()

# Preprocessing
df['month'] = pd.factorize(df['month'])[0]
df['day'] = pd.factorize(df['day'])[0]
df['fire'] = df['area'].apply(lambda x: 1 if x > 0 else 0)

X = df[['temp', 'RH', 'wind', 'rain']]
y = df['fire']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/forest_fire_model.pkl")
st.success("✅ Model trained and saved to 'model/forest_fire_model.pkl'")
