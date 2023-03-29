import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
from fredapi import Fred

# Get the closing stock prices
def get_stock_closing_prices(symbol, start_date, end_date):
    stock = yf.Ticker(symbol)
    hist = stock.history(start=start_date, end=end_date)
    hist.index = hist.index.tz_localize(None)  # Convert index to timezone-naive
    return hist['Close']

# Update the dataframe with the new earning dates
def find_next_earnings_date(row_date, earnings_dates):
    for ed in earnings_dates:
        if row_date < ed:
            return ed
    return None


# Replace 'your_api_key' with your FRED API key
fred = Fred(api_key='your_api_key')

def get_treasury_rate(start_date, end_date):
    rates = fred.get_series('DTB3', start_date, end_date) / 100
    rates.index = pd.to_datetime(rates.index).tz_localize(None)
    return rates


# Change Stock Symbol
symbol = 'TSLA'
stock = yf.Ticker(symbol)

#Get stock's Market Cap
market_cap = stock.fast_info['marketCap']

# Get Next Earnings Date
earnings_calendar = stock.get_earnings_dates(limit=21).index.values[4:]
next_earnings = earnings_calendar[0]
last_earnings = str(earnings_calendar[-1]).split('T')[0]
earnings_calendar = earnings_calendar[::-1]
format_string = "%Y-%m-%d"
dt = datetime.datetime.strptime(last_earnings, format_string)

end_date = datetime.date.today()
closing_prices = get_stock_closing_prices(symbol, dt, end_date)
df = pd.DataFrame({'Closing Price': closing_prices})

# Convert the earnings dates to datetime objects
earnings_dates = [pd.to_datetime(ed).to_pydatetime().date() for ed in earnings_calendar]

# Create the new column with the next earnings date
df['Next Earnings Date'] = df.index.map(lambda x: find_next_earnings_date(x.date(), earnings_dates))

# Get the S&P 500 closing prices
sp500_closing_prices = get_stock_closing_prices('^GSPC', dt, end_date)
sp500_df = pd.DataFrame({'S&P 500 Closing Price': sp500_closing_prices})

# Merge the two DataFrames on their indices
df = df.merge(sp500_df, left_index=True, right_index=True)

# Get the 3-month Treasury rate
treasury_rate = get_treasury_rate(dt, end_date)
treasury_df = pd.DataFrame({'3-Month Treasury Rate': treasury_rate})

# Merge the DataFrame
df = df.merge(treasury_df, left_index=True, right_index=True, how='left')

# Fill any missing Treasury rate data with the most recent available rate
df['3-Month Treasury Rate'].fillna(method='ffill', inplace=True)

# Download the stock_data csv
df.to_csv('stock_data.csv')

# Load the data
data = pd.read_csv('stock_data.csv')

# Convert 'Next Earnings Date' to datetime
data['Next Earnings Date'] = pd.to_datetime(data['Next Earnings Date'])

# Scale the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['Closing Price', 'S&P 500 Closing Price', '3-Month Treasury Rate']])

# Prepare the input for the LSTM model
def create_dataset(data, look_back=60):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), :])
        y.append(data[i + look_back, 0])  # 'Closing Price' is at index 0
    return np.array(X), np.array(y)

look_back = 60
X, y = create_dataset(scaled_data, look_back)

# Split the data into training and testing sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(1))

# Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Invert the scaling
y_train_pred = scaler.inverse_transform(np.concatenate((y_train_pred, X_train[:, -1, -2:]), axis=1))[:, 0]
y_test_pred = scaler.inverse_transform(np.concatenate((y_test_pred, X_test[:, -1, -2:]), axis=1))[:, 0]
y_train_actual = scaler.inverse_transform(np.concatenate((y_train.reshape(-1, 1), X_train[:, -1, -2:]), axis=1))[:, 0]
y_test_actual = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), X_test[:, -1, -2:]), axis=1))[:, 0]

# Calculate the mean squared error
print('Train MSE:', mean_squared_error(y_train_actual, y_train_pred))
print('Test MSE:', mean_squared_error(y_test_actual, y_test_pred))

# Plot the actual vs predicted stock prices
plt.figure(figsize=(14, 6))
plt.plot(y_test_actual, label='Actual Closing Price')
plt.plot(y_test_pred, label='Predicted Closing Price')
plt.xlabel('Time')
plt.ylabel('Closing Price')
plt.legend()
plt.show()

