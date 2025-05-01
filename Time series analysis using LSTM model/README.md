# Air Passengers Forecasting with LSTM

This project implements a Long Short-Term Memory (LSTM) neural network to forecast monthly international airline passengers using the classic **Air Passengers dataset** (1949–1960).

---

## Dataset
The dataset contains the total number of international airline passengers per month. It spans from January 1949 to December 1960.
File: `AirPassengers.csv`

---

## Model

The forecasting model uses an LSTM network with the following features:

- Input window: 12 months
- Model: Single-layer LSTM with 100 units
- Output: Prediction
- Scaled using `MinMaxScaler` from `sklearn`

---

## Requirements

Install the required Python libraries:

pip install numpy pandas matplotlib scikit-learn tensorflow

---

## Output
The script will:
  - Plot the historical data
  - Train the LSTM model using tanh activation function
  - Predict future passenger counts
  - Visualize actual vs. predicted values for both training and test sets
    
---

## Author
Dhanush B R
Deep Learning Project for Time Series Forecasting
