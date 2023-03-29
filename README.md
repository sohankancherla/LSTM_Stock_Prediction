# LSTM Stock Prediction
Automatically gathers stock data using a stock's symbol as a parameter. The program then trains a LSTM prediction model and displays the results from the test.

Tesla stock actual vs prediction:
![image](https://user-images.githubusercontent.com/30853467/228407717-ef0e6438-3346-4d7e-8627-0e42596e63db.png)

# How to Run
- replace fredapi with your own key (https://fred.stlouisfed.org/docs/api/fred/)
- chnage the stock variable to the stock symbol you want to predict
- cmd: python lstm_stock.py

# APIs and Libraries
- Fred API to get the 3 month treasury rate
- yfinance to get realtime stock data
- sci-kit learn to process and evalute the data
- keras to build the LSTM model
- matplotlib to plot the data
