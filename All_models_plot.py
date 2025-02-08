#!/usr/bin/env python
"""
Comparison of Three Forecasting Models for Natural Gas Prices:
1. SARIMA Model
2. Sine+Trend Model (Linear Trend + Sinusoidal Seasonality)
3. Prophet Model

This script loads historical monthly price data from a CSV file,
fits three different models, forecasts prices (extending 12 months into the future),
and produces graphs for each model.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
import statsmodels.api as sm
from scipy.interpolate import interp1d

# Import Prophet (if available)
try:
    from prophet import Prophet
except ImportError:
    print("Prophet library is not installed. Install it using 'pip install prophet'")
    sys.exit(1)

# ----------------------------
# Common Utility Functions
# ----------------------------
def load_data(csv_file: str) -> pd.DataFrame:
    """
    Load CSV data containing historical natural gas prices.
    The CSV file should have columns 'Dates' and 'Prices'.
    """
    if not os.path.isfile(csv_file):
        raise FileNotFoundError(f"CSV file '{csv_file}' not found.")
    try:
        df = pd.read_csv(csv_file, parse_dates=['Dates'])
    except Exception as e:
        raise ValueError(f"Error reading CSV file '{csv_file}': {e}")
    # Ensure data is sorted by date
    df.sort_values('Dates', inplace=True)
    return df

# ----------------------------
# SARIMA Model Functions
# ----------------------------
def fit_sarima_model(df: pd.DataFrame):
    """
    Fit a SARIMA model to the historical data.
    Uses SARIMA(1,1,1)x(1,1,1,12) as a starting point.
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

def forecast_future_prices(results, last_date: pd.Timestamp, steps: int = 12):
    """
    Forecast future prices using the fitted SARIMA model.
    Returns forecasted values and confidence intervals.
    """
    forecast_obj = results.get_forecast(steps=steps)
    forecast_index = [last_date + relativedelta(months=i) for i in range(1, steps + 1)]
    forecast_series = pd.Series(forecast_obj.predicted_mean, index=forecast_index)
    conf_int = forecast_obj.conf_int()
    conf_int.index = forecast_index
    return forecast_series, conf_int

def plot_sarima_forecast(df: pd.DataFrame, forecast_series: pd.Series, conf_int: pd.DataFrame):
    """
    Plot historical prices and SARIMA forecast with confidence intervals.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Prices'], marker='o', linestyle='-', label='Historical Prices')
    plt.plot(forecast_series.index, forecast_series, marker='o', linestyle='-', label='SARIMA Forecast')
    plt.fill_between(forecast_series.index,
                     conf_int.iloc[:, 0],
                     conf_int.iloc[:, 1],
                     color='pink', alpha=0.3, label='Confidence Interval')
    plt.xlabel("Dates")
    plt.ylabel("Prices")
    plt.title("SARIMA Model Forecast")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ----------------------------
# Sine+Trend Model Functions
# ----------------------------
def simple_regression(x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Perform a simple linear regression (y = slope * x + intercept).
    """
    xbar = np.mean(x)
    ybar = np.mean(y)
    slope = np.sum((x - xbar) * (y - ybar)) / np.sum((x - xbar)**2)
    intercept = ybar - slope * xbar
    return slope, intercept

def bilinear_regression(y: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> tuple:
    """
    Perform a bilinear regression without an intercept: y = slope1*x1 + slope2*x2.
    """
    slope1 = np.sum(y * x1) / np.sum(x1**2)
    slope2 = np.sum(y * x2) / np.sum(x2**2)
    return slope1, slope2

def fit_sine_trend_model(dates: list, prices: np.ndarray, start_date: date) -> dict:
    """
    Fit a model capturing a linear trend and a sinusoidal seasonal component.
    Returns a dictionary with model parameters.
    """
    # Convert dates to number of days from start_date
    days_from_start = np.array([(d - start_date).days for d in dates])
    # Fit linear trend
    slope, intercept = simple_regression(days_from_start, prices)
    # Detrend prices
    detrended = prices - (slope * days_from_start + intercept)
    # Define seasonal period (365 days)
    period = 365.0
    sin_comp = np.sin(2 * np.pi * days_from_start / period)
    cos_comp = np.cos(2 * np.pi * days_from_start / period)
    # Fit sine and cosine coefficients
    coeff_sin, coeff_cos = bilinear_regression(detrended, sin_comp, cos_comp)
    amplitude = np.sqrt(coeff_sin**2 + coeff_cos**2)
    shift = np.arctan2(coeff_cos, coeff_sin)
    return {
        'slope': slope,
        'intercept': intercept,
        'amplitude': amplitude,
        'shift': shift,
        'period': period
    }

def sine_trend_forecast(model_params: dict, start_date: date, end_date: date) -> (pd.DatetimeIndex, np.ndarray):
    """
    Generate a forecast using the sine+trend model from start_date to end_date.
    Returns a tuple of dates and estimated prices.
    """
    continuous_dates = pd.date_range(start=pd.Timestamp(start_date), end=pd.Timestamp(end_date), freq='D')
    estimated_prices = []
    for dt in continuous_dates:
        days = (dt.date() - start_date).days
        trend = model_params['slope'] * days + model_params['intercept']
        seasonal = model_params['amplitude'] * np.sin(2 * np.pi * days / model_params['period'] + model_params['shift'])
        estimated_prices.append(trend + seasonal)
    return continuous_dates, np.array(estimated_prices)

def plot_sine_trend_forecast(dates, prices, forecast_dates, forecast_prices):
    """
    Plot historical prices and the sine+trend model forecast.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dates, prices, 'o', label="Historical Prices")
    plt.plot(forecast_dates, forecast_prices, '-', label="Sine+Trend Forecast")
    plt.xlabel("Dates")
    plt.ylabel("Prices")
    plt.title("Sine+Trend Model Forecast")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ----------------------------
# Prophet Model Functions
# ----------------------------
def prepare_prophet_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for Prophet. Renames columns: 'Dates' -> 'ds', 'Prices' -> 'y'.
    """
    if 'Dates' not in df.columns or 'Prices' not in df.columns:
        raise ValueError("Data must contain 'Dates' and 'Prices' columns for Prophet.")
    prophet_df = df[['Dates', 'Prices']].rename(columns={'Dates': 'ds', 'Prices': 'y'})
    return prophet_df

def fit_prophet_model(prophet_df: pd.DataFrame) -> Prophet:
    """
    Fit a Prophet model to the data.
    """
    model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
    model.fit(prophet_df)
    return model

def forecast_prophet_model(model: Prophet, periods: int = 12, freq: str = 'M') -> pd.DataFrame:
    """
    Forecast future prices using Prophet.
    """
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast

def plot_prophet_forecast(model: Prophet, forecast: pd.DataFrame):
    """
    Plot the Prophet forecast and its components.
    """
    fig1 = model.plot(forecast)
    plt.title("Prophet Model Forecast")
    plt.xlabel("Dates")
    plt.ylabel("Prices")
    plt.show()
    
    fig2 = model.plot_components(forecast)
    plt.show()

# ----------------------------
# Main Execution Function
# ----------------------------
def main():
    # CSV file path (update if needed)
    csv_file = 'natural_gas_prices.csv'
    
    # Load data
    try:
        df = load_data(csv_file)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # For SARIMA, we set 'Date' as the index.
    df_sarima = df.copy()
    df_sarima.set_index('Dates', inplace=True)
    
    # ----------------------------
    # 1. SARIMA Model
    # ----------------------------
    print("Fitting SARIMA model...")
    try:
        sarima_results = fit_sarima_model(df_sarima)
        last_date = df_sarima.index.max()
        forecast_series, conf_int = forecast_future_prices(sarima_results, last_date, steps=12)
    except Exception as e:
        print(f"Error with SARIMA model: {e}")
        sys.exit(1)
    print("SARIMA model fitted and forecasted.")
    plot_sarima_forecast(df_sarima, forecast_series, conf_int)
    
    # ----------------------------
    # 2. Sine+Trend Model
    # ----------------------------
    print("Fitting Sine+Trend model...")
    # Use the original DataFrame for the sine+trend model.
    dates_list = df['Dates'].dt.date.tolist()
    prices = df['Prices'].values
    start_date_model = dates_list[0]  # Use first date as the reference
    try:
        model_params = fit_sine_trend_model(dates_list, prices, start_date_model)
    except Exception as e:
        print(f"Error fitting Sine+Trend model: {e}")
        sys.exit(1)
    print("Sine+Trend model parameters:", model_params)
    # Forecast from the first date up to one year beyond the last historical date.
    last_date_model = df['Dates'].max().date()
    forecast_end_date = last_date_model + relativedelta(months=12)
    forecast_dates, forecast_prices = sine_trend_forecast(model_params, start_date_model, forecast_end_date)
    plot_sine_trend_forecast(df['Dates'], prices, forecast_dates, forecast_prices)
    
    # ----------------------------
    # 3. Prophet Model
    # ----------------------------
    print("Fitting Prophet model...")
    try:
        prophet_df = prepare_prophet_data(df)
        model_prophet = fit_prophet_model(prophet_df)
        forecast_prophet = forecast_prophet_model(model_prophet, periods=12, freq='M')
    except Exception as e:
        print(f"Error with Prophet model: {e}")
        sys.exit(1)
    plot_prophet_forecast(model_prophet, forecast_prophet)

if __name__ == "__main__":
    main()
