import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

st.set_page_config(page_title="Forest Fire Detection", layout="centered")

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

# Train model if not already saved
model_path = "model/forest_fire_model.pkl"
if not os.path.exists(model_path):
    st.info("Training model...")
    X = df[['temp', 'RH', 'wind', 'rain']]
    y = df['fire']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, model_path)
    st.success("✅ Model trained and saved to 'model/forest_fire_model.pkl'")
else:
    model = joblib.load(model_path)

# Sidebar inputs for prediction
st.sidebar.header("📥 Input Features for Prediction")
temp = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, 25.0)
rh = st.sidebar.slider("Relative Humidity (%)", 0.0, 100.0, 45.0)
wind = st.sidebar.slider("Wind Speed (km/h)", 0.0, 20.0, 5.0)
rain = st.sidebar.slider("Rain (mm)", 0.0, 10.0, 0.0)

# Predict
input_data = pd.DataFrame([[temp, rh, wind, rain]], columns=['temp', 'RH', 'wind', 'rain'])
prediction = model.predict(input_data)[0]
prediction_label = "🔥 Fire Risk" if prediction == 1 else "✅ No Fire Risk"

st.subheader("📊 Prediction Result")
st.markdown(f"### {prediction_label}")

# Show input and prediction
st.write("#### Your Input:")
st.dataframe(input_data)

# Visualization
st.subheader("📈 Sample Data Visualization")
fire_counts = df['fire'].value_counts().rename({0: "No Fire", 1: "Fire"})
st.bar_chart(fire_counts)
