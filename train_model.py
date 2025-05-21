import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load dataset
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv")

# Preprocessing
df['month'] = pd.factorize(df['month'])[0]
df['day'] = pd.factorize(df['day'])[0]
df['fire'] = df['area'].apply(lambda x: 1 if x > 0 else 0)

# Features and target
X = df[['temp', 'RH', 'wind', 'rain']]
y = df['fire']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/forest_fire_model.pkl")
print("✅ Model saved successfully.")
