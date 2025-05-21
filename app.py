import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Forest Fire Predictor", layout="centered")
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

# Model path
model_path = "model/forest_fire_model.pkl"

# Train or load model
if not os.path.exists(model_path):
    st.warning("Training new model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, model_path)
    st.success("✅ Model trained and saved.")
else:
    try:
        model = joblib.load(model_path)
        st.success("✅ Model loaded successfully.")
    except:
        st.error("❌ Failed to load model. Delete the .pkl file and retrain.")

# Input form
st.sidebar.header("🧪 Input Environmental Conditions")

temp = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, 25.0)
rh = st.sidebar.slider("Relative Humidity (%)", 0.0, 100.0, 45.0)
wind = st.sidebar.slider("Wind Speed (km/h)", 0.0, 20.0, 5.0)
rain = st.sidebar.slider("Rain (mm)", 0.0, 10.0, 0.0)

if st.sidebar.button("Predict"):
    input_df = pd.DataFrame([[temp, rh, wind, rain]], columns=["temp", "RH", "wind", "rain"])
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    label = "🔥 Fire Risk Detected" if prediction == 1 else "✅ No Fire Risk"
    st.subheader("📢 Prediction Result")
    st.markdown(f"## {label}")
    st.write("### Input Data")
    st.dataframe(input_df)

    # Matplotlib visualization
    st.subheader("📊 Prediction Probability")
    fig, ax = plt.subplots()
    ax.bar(["No Fire", "Fire"], proba, color=["green", "red"])
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.set_title("Fire Risk Probability")
    st.pyplot(fig)
