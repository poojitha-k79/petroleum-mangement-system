import requests
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

API_KEY = "579b464db66ec23bdd0000016bc9e53abd4c447a6d682ac6fed61a95"
RESOURCE_ID = "6b608fff-7186-4d6b-bea4-e0d04a51a7ab"

url = f"https://api.data.gov.in/resource/{RESOURCE_ID}?api-key={API_KEY}&format=csv&limit=5000"

print("Fetching data from:", url)
response = requests.get(url)
response.raise_for_status()

# Load into DataFrame
df = pd.read_csv(StringIO(response.text))
print("✅ Data loaded. Shape:", df.shape)
print("Columns:", df.columns.tolist())

# Standardize column names
df.columns = df.columns.str.strip().str.upper()

# Convert numeric columns (Petrol/Diesel Prices)
for col in df.columns:
    if col not in ["COUNTRY"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna().reset_index(drop=True)

print("✅ Cleaned data. Shape:", df.shape)
print(df.head())

# Wide → Long format
long_df = df.melt(id_vars=["COUNTRY"],
                  value_vars=["PETROL PER LITRE", "DIESEL PER LITRE"],
                  var_name="PRODUCT", value_name="PRICE")

# Encode PRODUCT
long_df["PRODUCT_CODE"] = pd.factorize(long_df["PRODUCT"])[0]

print("✅ Reshaped dataset. Shape:", long_df.shape)
print(long_df.head())

# Let's predict Diesel price from Petrol price
petrol_data = df[["PETROL PER LITRE"]]
diesel_data = df["DIESEL PER LITRE"]

X_train, X_val, y_train, y_val = train_test_split(petrol_data, diesel_data, test_size=0.2, random_state=42)

model = XGBRegressor(objective="reg:squarederror", n_estimators=300, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

print("✅ ML Model trained.")
print("R² Score on Validation:", model.score(X_val, y_val))

# Predict example
example_petrol = np.array([[100]])  # example petrol price = 100
predicted_diesel = model.predict(example_petrol)
print(f"🔮 Predicted Diesel Price for Petrol=100: {predicted_diesel[0]:.2f}")

long_df.to_csv("cleaned_petroleum_data.csv", index=False)
print("💾 Saved to cleaned_petroleum_data.csv")
