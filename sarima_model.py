#!/usr/bin/env python
"""
JPMorgan Chase Natural Gas Price Analysis and Forecasting

Task Overview:
1. Load monthly natural gas price data (CSV) spanning from 31-Oct-2020 to 30-Sep-2024.
2. Visualize and analyze the data to identify patterns (seasonality, trends, etc.).
3. Fit a SARIMA model to the historical data.
4. Forecast prices for one extra year (12 months) beyond the last data point.
5. Provide a function that takes an input date and returns an estimated price.
   - If the date is within the historical data, use interpolation.
   - If the date is in the forecast window, interpolate the forecasted values.
   
Note: This solution uses a SARIMA model with parameters chosen as (1,1,1)x(1,1,1,12) – these
can be further tuned based on model diagnostics and the characteristics of your data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.interpolate import interp1d

# ------------------------------
# Data Loading and Preprocessing
# ------------------------------
def load_data(csv_file: str) -> pd.DataFrame:
    """
    Load the natural gas price CSV file, parse the Date column, set the index, and sort.
    Assumes the CSV has at least two columns: 'Dates' and 'Prices'.
    """
    try:
        df = pd.read_csv(csv_file, parse_dates=['Dates'])
    except Exception as e:
        raise IOError(f"Error reading {csv_file}: {e}")
    
    df.set_index('Dates', inplace=True)
    df.sort_index(inplace=True)
    return df

# ------------------------------
# Data Visualization Functions
# ------------------------------
def plot_time_series(df: pd.DataFrame):
    """
    Plot the historical time series data.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Prices'], marker='o', linestyle='-', label='Historical Prices')
    plt.title("Monthly Natural Gas Prices")
    plt.xlabel("Dates")
    plt.ylabel("Prices")
    plt.grid(True)
    plt.legend()
    plt.show()

def decompose_time_series(df: pd.DataFrame):
    """
    Decompose the time series into trend, seasonal, and residual components using additive decomposition.
    For monthly data, we use a period of 12.
    """
    result = seasonal_decompose(df['Prices'], model='additive', period=12)
    result.plot()
    plt.suptitle("Seasonal Decomposition of Natural Gas Prices", fontsize=14)
    plt.show()

# ------------------------------
# Modeling and Forecasting
# ------------------------------
def fit_sarima_model(df: pd.DataFrame):
    """
    Fit a SARIMA model to the historical data.
    The model chosen here is SARIMA(1, 1, 1)x(1, 1, 1, 12), which is often a good starting point
    for monthly data with seasonality. Adjust parameters as needed.
    """
    try:
        model = sm.tsa.statespace.SARIMAX(df['Prices'],
                                          order=(1, 1, 1),
                                          seasonal_order=(1, 1, 1, 12),
                                          enforce_stationarity=False,
                                          enforce_invertibility=False)
        results = model.fit(disp=False)
    except Exception as e:
        raise RuntimeError(f"Error fitting SARIMA model: {e}")
    return results

def forecast_future_prices(model_results, last_date: pd.Timestamp, steps: int = 12):
    """
    Forecast future prices for a specified number of months (steps).
    Returns:
        - forecast_series: A pandas Series indexed by forecast dates.
        - conf_int: The confidence intervals for the forecasts.
    """
    forecast_obj = model_results.get_forecast(steps=steps)
    forecast_index = [last_date + relativedelta(months=i) for i in range(1, steps + 1)]
    forecast_series = pd.Series(forecast_obj.predicted_mean, index=forecast_index)
    conf_int = forecast_obj.conf_int()
    conf_int.index = forecast_index
    return forecast_series, conf_int

def plot_forecast(df: pd.DataFrame, forecast_series: pd.Series, conf_int: pd.DataFrame):
    """
    Plot the historical data along with the forecasted prices and their confidence intervals.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Prices'], label='Historical Prices', marker='o')
    plt.plot(forecast_series.index, forecast_series, label='Forecast Prices', marker='o')
    plt.fill_between(forecast_series.index,
                     conf_int.iloc[:, 0],
                     conf_int.iloc[:, 1],
                     color='pink', alpha=0.3, label='Confidence Interval')
    plt.title("Natural Gas Prices Forecast")
    plt.xlabel("Dates")
    plt.ylabel("Prices")
    plt.legend()
    plt.grid(True)
    plt.show()

# ------------------------------
# Price Estimation Function
# ------------------------------
def get_price_estimate(input_date_str: str, df: pd.DataFrame, model_results) -> float:
    """
    Given an input date (as a string), return an estimated natural gas price.
    If the date is within the historical data, interpolate the value.
    If the date is within one year beyond the historical range, interpolate the forecast.
    
    Parameters:
        input_date_str (str): Date string in a recognizable format (e.g., "YYYY-MM-DD").
        df (pd.DataFrame): Historical data with a DateTimeIndex.
        model_results: Fitted SARIMA model results used for forecasting.
    
    Returns:
        float: Estimated price.
    
    Raises:
        ValueError: If the input date is not in a valid range or format.
    """
    try:
        input_date = pd.to_datetime(input_date_str)
    except Exception as e:
        raise ValueError("Invalid date format. Please provide a valid date string (e.g., 'YYYY-MM-DD').") from e

    historical_start = df.index.min()
    historical_end = df.index.max()
    forecast_end = historical_end + relativedelta(months=12)

    # Case 1: Input date is within historical data range
    if historical_start <= input_date <= historical_end:
        if input_date in df.index:
            # Exact date exists in historical data
            return float(df.loc[input_date, 'Prices'])
        else:
            # Use linear interpolation based on historical data
            # Convert dates to a numeric format (ordinal) for interpolation
            x_hist = np.array([d.toordinal() for d in df.index])
            y_hist = df['Prices'].values
            interp_func = interp1d(x_hist, y_hist, kind='linear', fill_value="extrapolate")
            return float(interp_func(input_date.toordinal()))
    
    # Case 2: Input date is in the forecast period (up to one year beyond the last historical date)
    elif historical_end < input_date <= forecast_end:
        # First, get the forecasted monthly values
        forecast_series, _ = forecast_future_prices(model_results, last_date=historical_end, steps=12)
        # For interpolation, convert forecast dates to ordinals
        x_forecast = np.array([d.toordinal() for d in forecast_series.index])
        y_forecast = forecast_series.values
        interp_func = interp1d(x_forecast, y_forecast, kind='linear', fill_value="extrapolate")
        return float(interp_func(input_date.toordinal()))
    else:
        raise ValueError("Date is out of allowed range. Provide a date within the historical data or up to one year beyond the last historical date.")

# ------------------------------
# Main Execution Block
# ------------------------------
if __name__ == "__main__":
    # Update the path to your CSV file accordingly
    csv_file = 'natural_gas_prices.csv'
    
    # Step 1: Load the data
    try:
        df = load_data(csv_file)
    except Exception as e:
        print(e)
        exit(1)
    
    # Step 2: Visualize the historical data
    print("Plotting historical data...")
    plot_time_series(df)
    
    # Step 3: Decompose the time series to check for seasonality and trends
    print("Performing seasonal decomposition...")
    decompose_time_series(df)
    
    # Step 4: Fit the SARIMA model
    print("Fitting SARIMA model. This may take a moment...")
    try:
        model_results = fit_sarima_model(df)
    except Exception as e:
        print(e)
        exit(1)
    print("Model fitted successfully.")
    
    # Step 5: Forecast future prices for one year (12 months)
    forecast_steps = 12
    forecast_series, conf_int = forecast_future_prices(model_results, last_date=df.index.max(), steps=forecast_steps)
    print("Forecast for the next 12 months completed.")
    
    # Step 6: Visualize the forecast along with historical data
    plot_forecast(df, forecast_series, conf_int)
    
    # Step 7: Take a date input from the user and provide a price estimate
    user_input_date = input("Enter a date (YYYY-MM-DD) for price estimation: ")
    try:
        estimated_price = get_price_estimate(user_input_date, df, model_results)
        print(f"Estimated natural gas price for {user_input_date} is: {estimated_price:.2f}")
    except Exception as e:
        print(f"Error: {e}")
