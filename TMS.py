import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Load clean data
cleaned_df = pd.read_csv("cleaned_petroleum_data.csv")

# Standardize column
cleaned_df.columns = cleaned_df.columns.str.strip().str.upper()

# Ensure columns exist
if "COUNTRY" not in cleaned_df.columns or "PRICE" not in cleaned_df.columns:
    raise ValueError("❌ Expected columns 'COUNTRY' and 'PRICE' not found. Check CSV headers.")

# Separate India and others
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

# Define trade action logic
def trade_decision(row, margin=5):
    if row["PRICE_DIFF"] > margin:
        return "SELL (Export)"   # other country pays more
    elif row["PRICE_DIFF"] < -margin:
        return "BUY (Import)"   # other country sells cheaper
    else:
        return "HOLD"           # difference too small

trade_df["TRADE_ACTION"] = trade_df.apply(trade_decision, axis=1)

# Model training (optional predictive step)
X = trade_df[["INDIA_BASE_PRICE", "PRODUCT_CODE"]]
y = trade_df["PRICE_DIFF"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=400, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_val)
print(f"✅ R² Score: {r2_score(y_val, preds):.3f}")
print(f"✅ MAE: {mean_absolute_error(y_val, preds):.3f}")

# Apply predicted differences for next trade decisions
trade_df["PREDICTED_DIFF"] = model.predict(X)
trade_df["PREDICTED_ACTION"] = trade_df["PREDICTED_DIFF"].apply(
    lambda x: "SELL (Export)" if x > 5 else ("BUY (Import)" if x < -5 else "HOLD")
)

# Show summary
print("\n🌍 Trade Management Summary (India as Base):")
print(trade_df[["COUNTRY", "PRODUCT", "PRICE", "INDIA_BASE_PRICE", "PRICE_DIFF", "TRADE_ACTION", "PREDICTED_ACTION"]].head(10))

# Save trade results
trade_df.to_csv("trade_management_predictions.csv", index=False)
print("\n💾 Results saved to 'trade_management_predictions.csv'")
