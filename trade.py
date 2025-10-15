import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# ============================
# 1️⃣ Load Cleaned Dataset
# ============================
df = pd.read_csv("cleaned_petroleum_data.csv")

print("✅ Data Loaded. Shape:", df.shape)
print(df.head())

# ============================
# 2️⃣ Data Preparation
# ============================

# Sort by country and product (simulate time trend)
df = df.sort_values(by=["COUNTRY", "PRODUCT_CODE"]).reset_index(drop=True)

# Create lag features (previous day prices)
df["PREV_PRICE"] = df.groupby("PRODUCT")["PRICE"].shift(1)
df["PRICE_CHANGE"] = df["PRICE"] - df["PREV_PRICE"]

# Drop first row per group where shift gives NaN
df = df.dropna().reset_index(drop=True)

# Feature and target
X = df[["PREV_PRICE", "PRODUCT_CODE", "PRICE_CHANGE"]]
y = df["PRICE"]

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ============================
# 3️⃣ Model Training
# ============================
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ============================
# 4️⃣ Model Evaluation
# ============================
y_pred = model.predict(X_val)
print("✅ Model trained successfully!")
print("R² Score:", r2_score(y_val, y_pred))
print("MAE:", mean_absolute_error(y_val, y_pred))

# ============================
# 5️⃣ Trade Decision Logic
# ============================
def trade_decision(current_price, predicted_price, threshold=0.5):
    """
    Decision logic:
    - Buy if price expected to rise more than threshold
    - Sell if price expected to fall more than threshold
    - Hold otherwise
    """
    change = predicted_price - current_price

    if change > threshold:
        decision = "BUY"
    elif change < -threshold:
        decision = "SELL"
    else:
        decision = "HOLD"

    profit_loss = change  # positive = profit, negative = loss
    return decision, profit_loss

# ============================
# 6️⃣ Example: Predict Trade for New Price
# ============================
product_code = 0  # 0 = Petrol, 1 = Diesel
current_price = 100
prev_price = 98
price_change = current_price - prev_price

example = np.array([[prev_price, product_code, price_change]])
predicted_next = model.predict(example)[0]

decision, profit_loss = trade_decision(current_price, predicted_next)

print("\n🔮 Predicted Next Price:", round(predicted_next, 2))
print("💡 Suggested Action:", decision)
print(f"💰 Expected Profit/Loss per litre: ₹{profit_loss:.2f}")

# ============================
# 7️⃣ Save Model
# ============================
joblib.dump(model, "trade_management_model.pkl")
print("💾 Model saved as trade_management_model.pkl")
