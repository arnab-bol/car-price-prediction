
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load dataset
df = pd.read_csv("data/car.csv")
print("Dataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

# 2. Clean dataset
df = df.drop_duplicates()
df["Selling_Price"] = pd.to_numeric(df["Selling_Price"], errors="coerce")
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df = df.dropna(subset=["Selling_Price", "Year"])

# 3. Feature engineering
REFERENCE_YEAR = pd.Timestamp.now().year
df["Car_Age"] = REFERENCE_YEAR - df["Year"]

# Car_Name is excluded to avoid unnecessary high-cardinality features
X = df[
    ["Present_Price", "Kms_Driven", "Fuel_Type",
     "Seller_Type", "Transmission", "Owner", "Car_Age"]
]
y = df["Selling_Price"]

numeric_features = [
    "Present_Price", "Kms_Driven", "Owner", "Car_Age"
]
categorical_features = [
    "Fuel_Type", "Seller_Type", "Transmission"
]

# 4. Preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# 6. Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    )
}

results = []
best_model = None
best_name = None
best_rmse = float("inf")
best_predictions = None

os.makedirs("models", exist_ok=True)

# 7. Train and evaluate
for name, regressor in models.items():
    model = Pipeline([
        ("preprocessing", preprocessor),
        ("regressor", regressor)
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    print(f"\n{name}")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2 Score: {r2:.3f}")

    if rmse < best_rmse:
        best_rmse = rmse
        best_model = model
        best_name = name
        best_predictions = predictions

# 8. Save model comparison and best model
results_df = pd.DataFrame(results).sort_values("RMSE")
results_df.to_csv("model_results.csv", index=False)
joblib.dump(best_model, "models/best_model.joblib")

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))
print("\nBest model:", best_name)
print("Saved: models/best_model.joblib")

# 9. Actual vs predicted graph
plt.figure(figsize=(8, 5))
plt.scatter(y_test, best_predictions, alpha=0.7)

low = min(y_test.min(), best_predictions.min())
high = max(y_test.max(), best_predictions.max())
plt.plot([low, high], [low, high], "r--")

plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title(f"Actual vs Predicted: {best_name}")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.show()

# 10. Feature importance for tree-based models
if best_name in ["Random Forest", "Gradient Boosting"]:
    regressor = best_model.named_steps["regressor"]
    names = best_model.named_steps[
        "preprocessing"
    ].get_feature_names_out()

    importance = pd.DataFrame({
        "Feature": names,
        "Importance": regressor.feature_importances_
    }).sort_values("Importance", ascending=False)

    print("\nTop features:\n", importance.head(10).to_string(index=False))

    top = importance.head(10).sort_values("Importance")
    plt.figure(figsize=(8, 5))
    plt.barh(top["Feature"], top["Importance"])
    plt.title("Car Price Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.show()

print("\nCar price prediction training completed!")
