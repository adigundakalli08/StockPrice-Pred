import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from SmartApi import SmartConnect 
from SearchScrip import get_nse_scrip_token
import pyotp
from logzero import logger
import config
import plotly.graph_objects as go

# Configure plotting style
plt.style.use('ggplot')

# Load API credentials
api_key = config.api_key
username = config.username
password = config.password

# Initialize SmartAPI
smartApi = SmartConnect(api_key)

try:
    token = config.token
    totp = pyotp.TOTP(token).now()
except Exception as e:
    logger.error("Invalid Token: The provided token is not valid.")
    raise e

# Generate session
data = smartApi.generateSession(username, password, totp)

if not data['status']:
    logger.error(data)
    raise Exception("Login failed")

authToken = data['data']['jwtToken']
refreshToken = data['data']['refreshToken']
feedToken = smartApi.getfeedToken()

# Fetch historical data
try:
    historicParam = {
        "exchange": "NSE",
        "symboltoken": get_nse_scrip_token("RELIANCE"),
        "interval": "ONE_DAY",
        "fromdate": "2019-01-01 00:00",
        "todate": "2024-04-19 00:00"
    }
    res = smartApi.getCandleData(historicParam)['data']

    df = pd.DataFrame.from_dict(res)
    df.columns = ['Date', 'open', 'high', 'low', 'close', 'volume']
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df.set_index('Date', inplace=True)

except Exception as e:
    logger.exception(f"Failed to fetch historical data: {e}")
    raise e

# Data preprocessing
df["HL_PCT"] = (df["high"] - df["close"]) / df["close"] * 100
df["PCT_Change"] = (df["close"] - df["open"]) / df["open"] * 100
df = df[["close", "HL_PCT", 'PCT_Change', "volume"]]

forecast_col = "close"
df.fillna(-9999, inplace=True)

forecast_out = int(np.ceil(0.01 * len(df)))
df["label"] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

# Prepare feature and label sets
X = np.array(df.drop(["label"], axis=1))
y = np.array(df["label"])

# Scale features
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
clf = LinearRegression().fit(X_train, y_train)

# Evaluate the model
accuracy = clf.score(X_test, y_test)
logger.info(f"Model Accuracy: {accuracy}")

# Predict future values
forecast_set = clf.predict(X_lately)

# Add forecast to dataframe
df["Forecast"] = np.NaN
last_date = df.iloc[-1].name
next_date = last_date + timedelta(days=1)

for i in forecast_set:
    next_date = next_date + timedelta(days=1)
    df.loc[next_date] = [np.NaN for _ in range(len(df.columns) - 1)] + [i]

# Plot the results
plt.figure(figsize=(10, 6))
df["close"].plot(label="Actual Price")
df["Forecast"].plot(label="Forecasted Price")
plt.legend(loc=4)
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Stock Price Prediction")
plt.show()
