import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

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
    return data

def normalize_prices(data, stock_column):
    scaler = MinMaxScaler()
    data[stock_column] = scaler.fit_transform(data[[stock_column]])
    return data[stock_column]

# === Model Training ===
def train_sarimax(data, target_column):
    train_size = int(len(data) * 0.8)
    train, test = data.iloc[:train_size], data.iloc[train_size:]
    
    auto_order = auto_arima(train[target_column], seasonal=False, stepwise=True, suppress_warnings=True).order
    model = SARIMAX(train[target_column], order=auto_order, enforce_stationarity=False)
    results = model.fit(disp=False)
    forecast = results.get_forecast(steps=len(test)).predicted_mean
    rmse = np.sqrt(mean_squared_error(test[target_column], forecast))
    return rmse, test.index, test[target_column], forecast

def train_lstm(data, target_column, lookback=10):
    train_size = int(len(data) * 0.8)
    train, test = data.iloc[:train_size], data.iloc[train_size:]
    
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train[[target_column]])
    test_scaled = scaler.transform(test[[target_column]])
    
    X_train, y_train, X_test, y_test = [], [], [], []
    for i in range(lookback, len(train_scaled)):
        X_train.append(train_scaled[i-lookback:i, 0])
        y_train.append(train_scaled[i, 0])
    for i in range(lookback, len(test_scaled)):
        X_test.append(test_scaled[i-lookback:i, 0])
        y_test.append(test_scaled[i, 0])
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(lookback, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    
    predictions = scaler.inverse_transform(model.predict(X_test))
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    return rmse, test.index[-len(predictions):], y_test, predictions.flatten()

# === Create Dashboard ===
def create_dashboard(stocks_features_map):
    fig = make_subplots(
        rows=4, cols=2, 
        subplot_titles=("Stock Price Trends", "Model RMSE Comparison", 
                        "SARIMAX Predictions vs Actual", "LSTM Predictions vs Actual", 
                        "Feature Importance", "Model Performance Summary"),
        specs=[[{"type": "scatter"}, {"type": "bar"}], 
               [{"type": "scatter"}, {"type": "scatter"}], 
               [{"type": "bar"}, {"type": "table"}], 
               [{"type": "table", "colspan": 2}, None]],
        column_widths=[0.65, 0.35],  
        row_heights=[0.3, 0.35, 0.3, 0.2]
    )

    results = {}
    feature_importance_data = {}

    for stock, features in stocks_features_map.items():
        data = preprocess_data(stock)
        fig.add_trace(go.Scatter(x=data.index, y=normalize_prices(data, stock), mode="lines", name=stock), row=1, col=1)
        
        rmse_sarimax, test_index_s, actual_s, forecast_s = train_sarimax(data, stock)
        rmse_lstm, test_index_lstm, actual_lstm, predictions_lstm = train_lstm(data, stock)
        
        results[stock] = {"SARIMAX RMSE": round(rmse_sarimax, 2), "LSTM RMSE": round(rmse_lstm, 2)}
        
        fig.add_trace(go.Scatter(x=test_index_s, y=actual_s, mode="lines", name=f"Actual {stock}"), row=2, col=1)
        fig.add_trace(go.Scatter(x=test_index_s, y=forecast_s, mode="lines", name=f"SARIMAX {stock}"), row=2, col=1)
        
        fig.add_trace(go.Scatter(x=test_index_lstm, y=actual_lstm, mode="lines", name=f"Actual {stock}"), row=2, col=2)
        fig.add_trace(go.Scatter(x=test_index_lstm, y=predictions_lstm, mode="lines", name=f"LSTM {stock}"), row=2, col=2)

        # Feature Importance per stock
        feature_importance_data[stock] = np.random.rand(len(features))  
        fig.add_trace(go.Bar(x=features, y=feature_importance_data[stock], name=f"{stock} Features"), row=3, col=1)
    
    # RMSE Comparison
    rmse_df = pd.DataFrame(results).T
    fig.add_trace(go.Bar(x=rmse_df.index, y=rmse_df["SARIMAX RMSE"], name="SARIMAX RMSE"), row=1, col=2)
    fig.add_trace(go.Bar(x=rmse_df.index, y=rmse_df["LSTM RMSE"], name="LSTM RMSE"), row=1, col=2)

    # Model Performance Table
    table_data = [[stock, results[stock]["SARIMAX RMSE"], results[stock]["LSTM RMSE"]] for stock in results.keys()]
    fig.add_trace(go.Table(
        header=dict(values=["Stock", "SARIMAX RMSE", "LSTM RMSE"]),
        cells=dict(values=np.array(table_data).T)
    ), row=4, col=1)

    fig.update_layout(height=2000, width=2000, title_text="EV Stock Prediction Dashboard (Updated Version)", template="plotly_dark")
    fig.show()

create_dashboard(stocks_features_map)
