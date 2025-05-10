import pandas as pd
import time
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import joblib
import numpy as np


start_time = time.time()

df = pd.read_csv('crop_yield.csv')


df = df.dropna()


label_cols = ['Region', 'Soil_Type', 'Crop', 'Weather_Condition']
encoders = {}
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le


df['Fertilizer_Used'] = df['Fertilizer_Used'].astype(int)
df['Irrigation_Used'] = df['Irrigation_Used'].astype(int)


X = df.drop('Yield_tons_per_hectare', axis=1)
y = df['Yield_tons_per_hectare']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost model and extended hyperparameter grid
xgb = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)

params = {
    'n_estimators': [100, 200, 500],
    'max_depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2],
    'alpha': [0, 0.1, 0.5],
}

# Randomized search with 5-fold CV and increased iterations for more thorough search
random_search = RandomizedSearchCV(estimator=xgb, param_distributions=params, n_iter=30, cv=5, scoring='r2', n_jobs=-1, verbose=1, random_state=42)


random_search.fit(X_train, y_train)


best_model = random_search.best_estimator_

# Evaluate model
y_pred = best_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n✅ XGBoost R² Score: {r2:.2f}")
print(f"📉 MAE: {mae:.4f}")
print(f"📉 RMSE: {rmse:.4f}")
print(f"🏅 Best Parameters: {random_search.best_params_}")


joblib.dump(best_model, 'xgb_crop_yield_model.pkl')
joblib.dump(encoders, 'label_encoders.pkl')
print("💾 Model and encoders saved successfully!")


end_time = time.time()
print(f"🕒 Training completed in {end_time - start_time:.2f} seconds")
