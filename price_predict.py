import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import gradio as gr
import matplotlib.pyplot as plt

# Example: fetched from your API previously
df = pd.DataFrame({
    'COUNTRY': ['India', 'USA', 'China', 'Germany'],
    'PETROL PER LITRE': [1.1, 0.8, 1.2, 1.5],
    'DIESEL PER LITRE': [0.9, 0.7, 1.0, 1.3]
})

# Standardize columns
df.columns = df.columns.str.strip().str.upper()

# Melt into long format
df_long = df.melt(id_vars=['COUNTRY'],
                  value_vars=['PETROL PER LITRE', 'DIESEL PER LITRE'],
                  var_name='PRODUCT',
                  value_name='PRICE')

# Clean PRODUCT names
df_long['PRODUCT'] = df_long['PRODUCT'].str.replace(' PER LITRE', '', regex=False)

# Create dummy DATE column
df_long['DATE'] = pd.date_range(start='2023-01-01', periods=len(df_long))

# Drop missing prices
df_long = df_long.dropna(subset=['PRICE'])

# Add feature columns for ML
df_long['PRODUCT_CODE'] = pd.factorize(df_long['PRODUCT'])[0]
df_long['COUNTRY_CODE'] = pd.factorize(df_long['COUNTRY'])[0]

# Simple lag feature
df_long = df_long.sort_values(['PRODUCT', 'DATE'])
df_long['LAG1'] = df_long.groupby('PRODUCT')['PRICE'].shift(1)

features = ['PRODUCT_CODE', 'COUNTRY_CODE', 'LAG1']
train_df = df_long.dropna(subset=['PRICE', 'LAG1'])

X = train_df[features]
y = train_df['PRICE']

model = XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.1, random_state=42)
model.fit(X, y)

def predict_price(country, product, lag_price):
    prod_code = pd.factorize(df_long['PRODUCT'])[0][df_long['PRODUCT'] == product][0]
    country_code = pd.factorize(df_long['COUNTRY'])[0][df_long['COUNTRY'] == country][0]
    X_new = pd.DataFrame([[prod_code, country_code, lag_price]], columns=features)
    pred = model.predict(X_new)[0]
    return round(pred, 2)

def plot_history(product):
    subset = df_long[df_long['PRODUCT'] == product]
    plt.figure(figsize=(6,4))
    plt.plot(subset['DATE'], subset['PRICE'], marker='o')
    plt.title(f'{product} Price History')
    plt.xlabel('Date')
    plt.ylabel('Price per litre')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return plt.gcf()

with gr.Blocks() as app:
    gr.Markdown("## Petroleum Product Management System")

    with gr.Tab("Price Prediction"):
        country = gr.Dropdown(choices=df_long['COUNTRY'].unique().tolist(), label="Select Country")
        product = gr.Dropdown(choices=df_long['PRODUCT'].unique().tolist(), label="Select Product")
        lag_price = gr.Number(label="Previous Price (Lag1)")
        btn = gr.Button("Predict Price")
        output = gr.Textbox(label="Predicted Price per litre")
        btn.click(predict_price, inputs=[country, product, lag_price], outputs=output)

    with gr.Tab("Historical Chart"):
        product_hist = gr.Dropdown(choices=df_long['PRODUCT'].unique().tolist(), label="Select Product")
        plot_btn = gr.Button("Show Price History")
        plot_output = gr.Plot()
        plot_btn.click(plot_history, inputs=[product_hist], outputs=plot_output)

app.launch()
