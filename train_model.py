import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
df = pd.read_csv("Preprocessed_Datathon_Data.csv")  
df = df.dropna(subset=["Predicted_Demand"])
df = df.drop(columns=["Date"], errors='ignore')
X = df.drop(columns=["Predicted_Demand"])
y = df["Predicted_Demand"]
X = pd.get_dummies(X)
joblib.dump(X.columns.tolist(), "columns.pkl")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor()
model.fit(X_train, y_train)
joblib.dump(model, "model.pkl")

print("✅ Model trained and saved as model.pkl, with columns.pkl for feature structure.")
