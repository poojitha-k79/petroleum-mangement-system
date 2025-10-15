import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Load clean data
cleaned_df = pd.read_csv("cleaned_petroleum_data.csv")

# Check structure
print("Columns:", cleaned_df.columns.tolist())
print("Sample:", cleaned_df.head())

# Filter base (India) vs other countries
india_data = cleaned_df[cleaned_df["COUNTRY"].str.contains("India", case=False, na=False)]
other_countries = cleaned_df[~cleaned_df["COUNTRY"].str.contains("India", case=False, na=False)]

if india_data.empty or other_countries.empty:
    raise ValueError("❌ No rows found for 'India' or other countries. Check COUNTRY column values.")

# Use average India petrol/diesel price as base
india_base_price = india_data["PRICE"].mean()

# Prepare trade dataset
trade_df = other_countries.copy()
trade_df["INDIA_BASE_PRICE"] = india_base_price
trade_df["PRICE_DIFF"] = trade_df["PRICE"] - india_base_price

print("✅ Trade dataset ready. Shape:", trade_df.shape)
print(trade_df.head())

# Features and Target
X = trade_df[["INDIA_BASE_PRICE", "PRODUCT_CODE"]]
y = trade_df["PRICE_DIFF"]

# Ensure not empty
if X.empty:
    raise ValueError("❌ Dataset is empty after filtering. Check input data again.")

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestRegressor(n_estimators=400, random_state=42)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_val)
print(f"✅ R² Score: {r2_score(y_val, preds):.3f}")
print(f"✅ MAE: {mean_absolute_error(y_val, preds):.3f}")

# Example prediction: Trade to UAE
example = pd.DataFrame({"INDIA_BASE_PRICE": [india_base_price], "PRODUCT_CODE": [1]})
predicted_diff = model.predict(example)[0]
predicted_price = india_base_price + predicted_diff

print(f"🌍 Predicted trade price for UAE (example): {predicted_price:.2f}")
