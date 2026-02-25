import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score
df = pd.read_csv("Preprocessed_Datathon_Data.csv")
X = df.drop(columns=['Daily_Sales_Quantity'])
y = df['Daily_Sales_Quantity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LGBMRegressor(
    n_estimators=150,
    learning_rate=0.08,
    num_leaves=40,
    max_depth=7,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("R²:", r2)
joblib.dump(model, "lgbm_model.pkl")
