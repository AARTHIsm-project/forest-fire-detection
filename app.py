import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Forest Fire Detection with Visualizations", layout="centered")
st.title("🔥 Forest Fire Detection System")

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv")
    df['month'] = pd.factorize(df['month'])[0]
    df['day'] = pd.factorize(df['day'])[0]
    df['fire'] = df['area'].apply(lambda x: 1 if x > 0 else 0)
    return df

df = load_data()
X = df[['temp', 'RH', 'wind', 'rain']]
y = df['fire']

model_path = "model/forest_fire_model.pkl"

def train_and_save_model():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, model_path)
    return model

try:
    model = joblib.load(model_path)
    st.success("✅ Model loaded successfully.")
except Exception as e:
    st.warning(f"Model load failed: {e}\n Retraining model...")
    if os.path.exists(model_path):
        os.remove(model_path)
    model = train_and_save_model()
    st.success("✅ Model trained and saved successfully.")

# Sidebar inputs
st.sidebar.header("🧪 Input Environmental Conditions")
temp = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, 25.0)
rh = st.sidebar.slider("Relative Humidity (%)", 0.0, 100.0, 45.0)
wind = st.sidebar.slider("Wind Speed (km/h)", 0.0, 20.0, 5.0)
rain = st.sidebar.slider("Rain (mm)", 0.0, 10.0, 0.0)

if st.sidebar.button("Predict"):
    input_df = pd.DataFrame([[temp, rh, wind, rain]], columns=['temp', 'RH', 'wind', 'rain'])
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    label = "🔥 Fire Risk Detected" if prediction == 1 else "✅ No Fire Risk"
    st.subheader("📢 Prediction Result")
    st.markdown(f"## {label}")
    st.write("### Input Data")
    st.dataframe(input_df)

    # Bar chart of probabilities
    fig1, ax1 = plt.subplots()
    ax1.bar(["No Fire", "Fire"], proba, color=["green", "red"])
    ax1.set_ylabel("Probability")
    ax1.set_ylim(0, 1)
    ax1.set_title("Fire Risk Probability")
    st.pyplot(fig1)

# --- Visualization Section ---
st.header("📊 Data Visualizations")

# 1. Heatmap: Correlation matrix
st.subheader("Heatmap: Feature Correlations")
fig2, ax2 = plt.subplots(figsize=(6,5))
corr = df[['temp', 'RH', 'wind', 'rain', 'fire']].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax2)
st.pyplot(fig2)

# 2. Line plot: Avg temp & humidity by month
st.subheader("Line Plot: Average Temperature & Humidity by Month")
monthly_avg = df.groupby('month')[['temp', 'RH']].mean()
fig3, ax3 = plt.subplots()
ax3.plot(monthly_avg.index, monthly_avg['temp'], label='Avg Temp (°C)', marker='o')
ax3.plot(monthly_avg.index, monthly_avg['RH'], label='Avg RH (%)', marker='s')
ax3.set_xlabel("Month (encoded)")
ax3.set_ylabel("Value")
ax3.set_title("Average Temperature and Humidity by Month")
ax3.legend()
st.pyplot(fig3)

# 3. Bar chart: Fire counts
st.subheader("Bar Chart: Fire vs No Fire Counts")
fire_counts = df['fire'].value_counts().rename({0:"No Fire", 1:"Fire"})
fig4, ax4 = plt.subplots()
ax4.bar(fire_counts.index, fire_counts.values, color=['green', 'red'])
ax4.set_xticks([0,1])
ax4.set_xticklabels(["No Fire", "Fire"])
ax4.set_ylabel("Counts")
ax4.set_title("Fire vs No Fire Counts")
st.pyplot(fig4)
