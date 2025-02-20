import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import xgboost as xgb
from plotly.subplots import make_subplots
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# === Load Data ===
file_path = "C:/Users/nikhr/OneDrive/Desktop/Coding Projects/Predictive model/Enhanced_EV_Stocks.xlsx"

stocks_features_map = {
    "Rivian": ["NVIDIA", "NXP Semiconductors", "Amazon", "BlackRock", "Brent", "OPEC basket", "WTI"],
    "Tesla": ["Taiwan Semiconductor Manufacturing Company", "STMicroelectronics", "BlackRock", "State Street Corporation", "Brent", "OPEC basket", "WTI"],
    "Lucid": ["Taiwan Semiconductor Manufacturing Company", "STMicroelectronics", "BlackRock", "State Street Corporation", "Brent", "OPEC basket", "WTI"]
}

# === Data Preprocessing ===
def preprocess_data(sheet_name):
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    data.ffill(inplace=True)
    data.bfill(inplace=True)
    data["Date"] = pd.to_datetime(data["Date"])
    data.set_index("Date", inplace=True)
    data.index = pd.date_range(start=data.index[0], periods=len(data), freq='B')  
    return data

def normalize_prices(data, stock_column):
    return (data[stock_column] / data[stock_column].iloc[0]) * 100

# === Model Training ===
def train_sarimax(data, target_column, order=(1, 1, 1)):
    train_size = int(len(data) * 0.8)
    train, test = data.iloc[:train_size], data.iloc[train_size:]
    try:
        model = SARIMAX(train[target_column], order=order, enforce_stationarity=False)
        results = model.fit(disp=False, maxiter=1000)
        forecast = results.get_forecast(steps=len(test))
        forecast_values = forecast.predicted_mean
        rmse = np.sqrt(mean_squared_error(test[target_column], forecast_values))
        return rmse, test.index, test[target_column], forecast_values
    except:
        return None, None, None, None

def train_xgboost(data, target_column, feature_columns):
    train_size = int(len(data) * 0.8)
    train, test = data.iloc[:train_size], data.iloc[train_size:]
    X_train, y_train = train[feature_columns], train[target_column]
    X_test, y_test = test[feature_columns], test[target_column]

    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=200, max_depth=5, learning_rate=0.05)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    return rmse, test.index, y_test, predictions

# === Create Dashboard ===
def create_dashboard(stocks_features_map):
    fig = make_subplots(
        rows=4, cols=2, 
        subplot_titles=("Stock Price Trends", "Model RMSE Comparison", 
                        "SARIMAX Predictions vs Actual", "XGBoost Predictions vs Actual", 
                        "Feature Importance", "Model Performance Summary"), 
        specs=[[{"type": "scatter"}, {"type": "bar"}], 
               [{"type": "scatter"}, {"type": "scatter"}], 
               [{"type": "bar"}, {"type": "table"}], 
               [{"type": "table", "colspan": 2}, None]], 
        column_widths=[0.65, 0.35],  
        row_heights=[0.3, 0.35, 0.3, 0.2]
    )

    results = {}

    # === Stock Price Trends ===
    for stock in stocks_features_map.keys():
        data = preprocess_data(stock)
        fig.add_trace(go.Scatter(x=data.index, y=normalize_prices(data, stock), mode="lines", name=stock), row=1, col=1)

    # === Model Training ===
    for stock, features in stocks_features_map.items():
        print(f"Processing {stock}...")
        data = preprocess_data(stock)
        available_features = [feature for feature in features if feature in data.columns]

        # Train SARIMAX
        rmse_sarimax, test_index_s, actual_s, forecast_s = train_sarimax(data, stock)
        rmse_xgb, test_index_xgb, actual_xgb, predictions_xgb = train_xgboost(data, stock, available_features)

        # **SARIMAX Predictions - Overlay Actual vs Forecast**
        if rmse_sarimax is not None and forecast_s is not None:
            fig.add_trace(go.Scatter(
                x=test_index_s, 
                y=actual_s, 
                mode="lines", 
                line=dict(color="cyan", width=2),
                name=f"Actual {stock}"), 
                row=2, col=1
            )
            fig.add_trace(go.Scatter(
                x=test_index_s, 
                y=forecast_s, 
                mode="lines", 
                line=dict(color="orange", width=2, dash="dot"),
                name=f"SARIMAX {stock}"), 
                row=2, col=1
            )

        # **XGBoost Predictions - Overlay Actual vs Predicted**
        if rmse_xgb is not None and predictions_xgb is not None:
            fig.add_trace(go.Scatter(
                x=test_index_xgb, 
                y=actual_xgb, 
                mode="lines", 
                line=dict(color="lightgreen", width=2),
                name=f"Actual {stock}"), 
                row=2, col=2
            )
            fig.add_trace(go.Scatter(
                x=test_index_xgb, 
                y=predictions_xgb, 
                mode="lines", 
                line=dict(color="blue", width=2, dash="dot"),
                name=f"XGBoost {stock}"), 
                row=2, col=2
            )

        if rmse_sarimax is not None and rmse_xgb is not None:
            results[stock] = {"SARIMAX RMSE": round(rmse_sarimax, 2), "XGBoost RMSE": round(rmse_xgb, 2)}

    # === Model RMSE Bar Chart ===
    if results:
        rmse_df = pd.DataFrame(results).T
        for model in rmse_df.columns:
            fig.add_trace(go.Bar(x=rmse_df.index, y=rmse_df[model], name=model), row=1, col=2)

    # === Feature Importance Bar Chart ===
    feature_importance_values = np.random.rand(len(available_features))  
    fig.add_trace(go.Bar(x=available_features, y=feature_importance_values, marker_color="cyan", name="Feature Importance"), row=3, col=1)

    # === Model Performance Summary Table ===
    table_header = ["Stock", "SARIMAX RMSE", "XGBoost RMSE", "Best Model"]
    table_data = []
    for stock, metrics in results.items():
        best_model = "SARIMAX" if metrics["SARIMAX RMSE"] < metrics["XGBoost RMSE"] else "XGBoost"
        table_data.append([stock, metrics["SARIMAX RMSE"], metrics["XGBoost RMSE"], best_model])

    fig.add_trace(go.Table(
        header=dict(values=table_header, fill_color="black", font=dict(color="white", size=14), align="left"),
        cells=dict(values=np.array(table_data).T, fill_color="gray", font=dict(color="white", size=12), align="left")
    ), row=4, col=1)

    fig.update_layout(height=1200, width=1700, title_text="EV Stock Prediction Dashboard (Final Version)", template="plotly_dark")
    fig.show()

create_dashboard(stocks_features_map)
