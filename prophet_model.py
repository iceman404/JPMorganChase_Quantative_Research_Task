#!/usr/bin/env python
"""
Natural Gas Price Forecasting using Prophet

This script demonstrates the use of the Prophet model to forecast natural gas prices.
It loads historical monthly price data from a CSV file, fits a Prophet model,
forecasts prices for one year beyond the latest historical date, and provides
an interactive function to get a price estimate for any given date.

Requirements:
    - prophet (install via: pip install prophet)
    - pandas
    - numpy
    - matplotlib
    - python-dateutil
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# Try to import Prophet from the prophet library
try:
    from prophet import Prophet
except ImportError as e:
    raise ImportError("Prophet library is not installed. Please install it using 'pip install prophet'") from e

# -------------------------------------------------------------------
# Data Loading and Preprocessing Functions
# -------------------------------------------------------------------
def load_data(csv_file: str) -> pd.DataFrame:
    """
    Load CSV data containing historical natural gas prices.

    The CSV file should have at least two columns:
        - 'Dates': Date strings (e.g., '2020-10-31')
        - 'Prices': Natural gas price values

    Parameters:
        csv_file (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame with parsed dates.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If an error occurs while reading the CSV.
    """
    if not os.path.isfile(csv_file):
        raise FileNotFoundError(f"CSV file '{csv_file}' not found.")
    try:
        df = pd.read_csv(csv_file, parse_dates=['Dates'])
    except Exception as e:
        raise ValueError(f"Error reading CSV file '{csv_file}': {e}")
    
    # Sort the data by date to ensure chronological order
    df.sort_values('Dates', inplace=True)
    return df

def prepare_prophet_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the data for Prophet modeling.

    Prophet expects two columns:
        - 'ds': The date column.
        - 'y': The target variable (price).

    Parameters:
        df (pd.DataFrame): Original DataFrame with columns 'Date' and 'Price'.

    Returns:
        pd.DataFrame: DataFrame with columns renamed for Prophet.

    Raises:
        ValueError: If required columns are missing.
    """
    # Ensure that the required columns exist in the dataframe
    if 'Dates' not in df.columns or 'Prices' not in df.columns:
        raise ValueError("CSV data must contain 'Dates' and 'Prices' columns.")
    
    # Rename columns to 'ds' and 'y' as required by Prophet
    prophet_df = df[['Dates', 'Prices']].rename(columns={'Dates': 'ds', 'Prices': 'y'})
    return prophet_df

# -------------------------------------------------------------------
# Prophet Modeling Functions
# -------------------------------------------------------------------
def fit_prophet_model(prophet_df: pd.DataFrame) -> Prophet:
    """
    Fit a Prophet model to the historical data.

    Parameters:
        prophet_df (pd.DataFrame): DataFrame prepared for Prophet with columns 'ds' and 'y'.

    Returns:
        Prophet: Fitted Prophet model.
    """
    # Initialize the Prophet model
    # Enabling yearly seasonality; disable daily and weekly seasonality since we have monthly data.
    model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
    
    # Fit the model to the data
    model.fit(prophet_df)
    return model

def forecast_future_prices(model: Prophet, periods: int = 12) -> pd.DataFrame:
    """
    Forecast future natural gas prices using the fitted Prophet model.

    Parameters:
        model (Prophet): Fitted Prophet model.
        periods (int): Number of future monthly periods to forecast (default: 12).

    Returns:
        pd.DataFrame: Forecasted results with predictions and confidence intervals.
    """
    # Get the last date from the historical data used by Prophet
    last_date = model.history['ds'].max()
    
    # Create future dates: since our data is monthly, we create a date range with monthly frequency.
    future_dates = pd.date_range(start=last_date + relativedelta(days=1), periods=periods, freq='M')
    future = pd.DataFrame({'ds': future_dates})
    
    # Make predictions on the future dates
    forecast = model.predict(future)
    return forecast

def plot_prophet_forecast(model: Prophet, forecast: pd.DataFrame, prophet_df: pd.DataFrame) -> None:
    """
    Plot the forecasted prices using Prophet's built-in plotting functions.

    Parameters:
        model (Prophet): Fitted Prophet model.
        forecast (pd.DataFrame): Forecasted DataFrame from Prophet.
        prophet_df (pd.DataFrame): Original data used for modeling.
    """
    # Plot the forecasted values along with historical data
    fig1 = model.plot(forecast)
    plt.title("Prophet Forecast of Natural Gas Prices")
    plt.xlabel("Dates")
    plt.ylabel("Prices")
    plt.show()
    
    # Plot forecast components (trend, yearly seasonality, etc.)
    fig2 = model.plot_components(forecast)
    plt.show()

def get_price_estimate_prophet(query_date_str: str, model: Prophet) -> float:
    """
    Estimate the natural gas price for a given date using the Prophet model.

    Parameters:
        query_date_str (str): Date string (format: 'YYYY-MM-DD') for which to estimate the price.
        model (Prophet): Fitted Prophet model.

    Returns:
        float: Estimated natural gas price.

    Raises:
        ValueError: If the date format is invalid.
    """
    try:
        # Convert the query date string to a datetime object
        query_date = pd.to_datetime(query_date_str)
    except Exception as e:
        raise ValueError("Invalid date format. Please use 'YYYY-MM-DD'.") from e
    
    # Create a DataFrame for the query date for prediction
    query_df = pd.DataFrame({'ds': [query_date]})
    forecast = model.predict(query_df)
    
    # 'yhat' is the point forecast; additional columns provide uncertainty intervals
    price_estimate = forecast['yhat'].iloc[0]
    return price_estimate

# -------------------------------------------------------------------
# Main Execution Function
# -------------------------------------------------------------------
def main():
    """
    Main function to execute the Prophet-based natural gas price forecasting.
    """
    # Specify the CSV file path (update as needed)
    csv_file = 'natural_gas_prices.csv'
    
    # Step 1: Load the data
    try:
        df = load_data(csv_file)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Step 2: Prepare data for Prophet (columns: 'ds' and 'y')
    try:
        prophet_df = prepare_prophet_data(df)
    except Exception as e:
        print(f"Error preparing data for Prophet: {e}")
        sys.exit(1)
    
    # Step 3: Fit the Prophet model
    print("Fitting the Prophet model to historical data...")
    try:
        model = fit_prophet_model(prophet_df)
    except Exception as e:
        print(f"Error fitting Prophet model: {e}")
        sys.exit(1)
    print("Model fitted successfully.")
    
    # Step 4: Forecast future prices for the next 12 months
    try:
        forecast = forecast_future_prices(model, periods=12)
    except Exception as e:
        print(f"Error forecasting future prices: {e}")
        sys.exit(1)
    
    # Step 5: Plot the forecast and its components
    plot_prophet_forecast(model, forecast, prophet_df)
    
    # Step 6: Interactive prompt for price estimation on a specific date
    input_date_str = input("Enter a date (YYYY-MM-DD) for price estimation: ")
    try:
        estimated_price = get_price_estimate_prophet(input_date_str, model)
        print(f"Estimated natural gas price for {input_date_str} is: {estimated_price:.2f}")
    except Exception as e:
        print(f"Error estimating price: {e}")

if __name__ == "__main__":
    main()
