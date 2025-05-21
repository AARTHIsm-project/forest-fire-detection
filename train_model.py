import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("data/forestfires.csv")

df['month'] = pd.factorize(df['month'])[0]
df['day'] = pd.factorize(df['day'])[0]

df['fire'] = df['area'].apply(lambda x: 1 if x > 0 else 0)

X = df[['temp', 'RH', 'wind', 'rain']]
y = df['fire']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

joblib.dump(model, 'model/forest_fire_model.pkl')
print("✅ Model saved to model/forest_fire_model.pkl")
