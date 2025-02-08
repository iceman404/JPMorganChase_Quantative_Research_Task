#!/usr/bin/env python
"""
Industry-Standard Implementation of Natural Gas Price Analysis Using
a Linear Trend + Sinusoidal Seasonal Model

This code is a refactored version of the official sample answer.
It uses modular functions, robust error handling, and extensive comments.
It reads monthly natural gas price data from a CSV, fits a model that
captures both the linear trend and the seasonal (sinusoidal) component,
and provides an interpolation function for price estimation.

Assumptions:
    - The CSV file (default: 'natural_gas_prices.csv') has at least two columns:
      'Dates' and 'Prices'.
    - The dates in the CSV represent the monthly snapshot dates.
    - The dataset spans from October 31, 2020 to September 30, 2024.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta

# -------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------
def change_working_directory(new_directory: str) -> None:
    """
    Change the current working directory to the provided path.
    
    Parameters:
        new_directory (str): Target directory path.
        
    Raises:
        FileNotFoundError: If the new_directory does not exist.
    """
    if not os.path.isdir(new_directory):
        raise FileNotFoundError(f"Directory '{new_directory}' does not exist.")
    os.chdir(new_directory)
    print(f"Changed working directory to: {os.getcwd()}")

def load_csv_data(filepath: str, date_column: str = 'Dates') -> pd.DataFrame:
    """
    Load CSV data from a given filepath, parsing the specified date column.
    
    Parameters:
        filepath (str): Path to the CSV file.
        date_column (str): Name of the date column to parse.
        
    Returns:
        pd.DataFrame: DataFrame with parsed dates.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If an error occurs while reading the CSV.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"CSV file '{filepath}' does not exist.")
    try:
        df = pd.read_csv(filepath, parse_dates=[date_column])
    except Exception as e:
        raise ValueError(f"Error reading CSV file '{filepath}': {e}")
    return df

# -------------------------------------------------------------------
# Data Processing Functions
# -------------------------------------------------------------------
def compute_monthly_dates(start_date: date, end_date: date) -> list:
    """
    Generate a list of monthly dates representing the last day of each month,
    from start_date up to end_date.
    
    Parameters:
        start_date (date): The start date.
        end_date (date): The end date.
    
    Returns:
        list: A list of date objects for the end of each month.
    """
    monthly_dates = []
    current_date = start_date
    while current_date <= end_date:
        # Calculate the last day of the current month.
        next_month = current_date.replace(day=28) + timedelta(days=4)
        last_day = next_month - timedelta(days=next_month.day)
        monthly_dates.append(last_day)
        # Advance to the first day of the next month.
        current_date = last_day + timedelta(days=1)
    return monthly_dates

def simple_regression(x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Perform a simple linear regression (y = slope * x + intercept).
    
    Parameters:
        x (np.ndarray): Independent variable.
        y (np.ndarray): Dependent variable.
        
    Returns:
        tuple: (slope, intercept)
    """
    xbar = np.mean(x)
    ybar = np.mean(y)
    slope = np.sum((x - xbar) * (y - ybar)) / np.sum((x - xbar) ** 2)
    intercept = ybar - slope * xbar
    return slope, intercept

def bilinear_regression(y: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> tuple:
    """
    Perform a bilinear regression without intercept to estimate coefficients
    for two predictors in the model: y = slope1 * x1 + slope2 * x2.
    
    Parameters:
        y (np.ndarray): Dependent variable.
        x1 (np.ndarray): First predictor.
        x2 (np.ndarray): Second predictor.
        
    Returns:
        tuple: (slope1, slope2)
    """
    slope1 = np.sum(y * x1) / np.sum(x1 ** 2)
    slope2 = np.sum(y * x2) / np.sum(x2 ** 2)
    return slope1, slope2

def fit_trend_and_seasonality(days_from_start: np.ndarray, prices: np.ndarray) -> dict:
    """
    Fit a model that captures a linear trend and a sinusoidal seasonal component.
    The model is defined as:
        price = (slope * days + intercept) + amplitude * sin(2*pi/period * days + shift)
    
    Parameters:
        days_from_start (np.ndarray): Days elapsed from the start date.
        prices (np.ndarray): Price values.
        
    Returns:
        dict: A dictionary with model parameters (slope, intercept, amplitude, shift, period).
    """
    # Fit the linear trend using simple regression.
    slope, intercept = simple_regression(days_from_start, prices)
    
    # Detrend the data to isolate seasonal variations.
    detrended_prices = prices - (slope * days_from_start + intercept)
    
    # Define the period (365 days for annual seasonality).
    period = 365.0
    sin_component = np.sin(2 * np.pi * days_from_start / period)
    cos_component = np.cos(2 * np.pi * days_from_start / period)
    
    # Fit the sine and cosine components using bilinear regression.
    coeff_sin, coeff_cos = bilinear_regression(detrended_prices, sin_component, cos_component)
    
    # Compute amplitude and phase shift.
    amplitude = np.sqrt(coeff_sin**2 + coeff_cos**2)
    shift = np.arctan2(coeff_cos, coeff_sin)
    
    return {
        'slope': slope,
        'intercept': intercept,
        'amplitude': amplitude,
        'shift': shift,
        'period': period
    }

def model_price_estimate(days: int, model_params: dict) -> float:
    """
    Estimate the natural gas price at a given number of days from the start date,
    based on the fitted linear trend and seasonal (sinusoidal) model.
    
    Parameters:
        days (int): Number of days elapsed from the start date.
        model_params (dict): Dictionary containing model parameters.
        
    Returns:
        float: Estimated price.
    """
    trend = model_params['slope'] * days + model_params['intercept']
    seasonal = model_params['amplitude'] * np.sin(2 * np.pi * days / model_params['period'] + model_params['shift'])
    return trend + seasonal

def interpolate_price(query_date: pd.Timestamp, start_date: pd.Timestamp, 
                        known_dates: list, known_prices: np.ndarray, model_params: dict) -> float:
    """
    Estimate the natural gas price for a given query date. If the query date matches a known date,
    return the exact price; otherwise, use the fitted trend + seasonality model to interpolate/extrapolate.
    
    Parameters:
        query_date (pd.Timestamp): The date for which to estimate the price.
        start_date (pd.Timestamp): The reference start date of the dataset.
        known_dates (list): List of known snapshot dates (as date objects).
        known_prices (np.ndarray): Array of known price values.
        model_params (dict): Dictionary of the fitted model parameters.
        
    Returns:
        float: Estimated price.
        
    Raises:
        ValueError: If the query_date is before the start_date.
    """
    if query_date < start_date:
        raise ValueError("Query date cannot be before the start date of the dataset.")
    
    # Calculate the number of days from start_date.
    days = (query_date - start_date).days
    
    # Check if the query_date exactly matches any known snapshot.
    for idx, known_date in enumerate(known_dates):
        if query_date.date() == known_date:
            return known_prices[idx]
    
    # Use the model for interpolation/extrapolation.
    return model_price_estimate(days, model_params)

# -------------------------------------------------------------------
# Visualization Functions
# -------------------------------------------------------------------
def plot_prices_and_fit(known_dates: list, known_prices: np.ndarray, start_date: pd.Timestamp, 
                        model_params: dict) -> None:
    """
    Plot the monthly natural gas prices along with the continuous trend + seasonal fit.
    
    Parameters:
        known_dates (list): List of known snapshot dates.
        known_prices (np.ndarray): Known price values.
        start_date (pd.Timestamp): Reference start date.
        model_params (dict): Fitted model parameters.
    """
    # Create a continuous date range from start_date to the last known date.
    end_date = known_dates[-1]
    continuous_dates = pd.date_range(start=start_date, end=pd.Timestamp(end_date), freq='D')
    
    # Compute the model estimates for each day in the continuous range.
    estimated_prices = [
        model_price_estimate((dt - start_date).days, model_params) for dt in continuous_dates
    ]
    
    plt.figure(figsize=(12, 6))
    plt.plot(known_dates, known_prices, 'o', label='Monthly Prices')
    plt.plot(continuous_dates, estimated_prices, '-', label='Trend + Seasonal Fit')
    plt.xlabel('Date')
    plt.ylabel('Natural Gas Price')
    plt.title('Natural Gas Prices with Fitted Trend and Seasonality')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------
# Main Execution Function
# -------------------------------------------------------------------
def main():
    """
    Main function to execute the analysis:
        - Loads CSV data.
        - Processes the data and computes monthly snapshot dates.
        - Fits a model for the linear trend and seasonal component.
        - Plots the fitted model against the original data.
        - Provides an interactive prompt for price estimation.
    """
    # Optional: Change the working directory if needed.
    # new_directory = "path_to_your_directory"
    # try:
    #     change_working_directory(new_directory)
    # except Exception as e:
    #     print(f"Error changing directory: {e}")
    #     sys.exit(1)
    
    # Specify the CSV file path.
    csv_filepath = 'natural_gas_prices.csv'
    
    # Load CSV data.
    try:
        df = load_csv_data(csv_filepath, date_column='Dates')
    except Exception as e:
        print(f"Error loading CSV data: {e}")
        sys.exit(1)
    
    # Ensure the 'Prices' column exists.
    if 'Prices' not in df.columns:
        print("Error: 'Prices' column not found in CSV data.")
        sys.exit(1)
    
    # Sort the data by dates.
    df.sort_values(by='Dates', inplace=True)
    
    # Extract known dates and prices.
    known_dates = df['Dates'].dt.date.tolist()  # Convert to date objects
    known_prices = df['Prices'].values
    
    # Define the dataset range (as per the task).
    start_date = date(2020, 10, 31)
    end_date = date(2024, 9, 30)
    
    # Compute monthly dates based on start and end dates.
    monthly_dates = compute_monthly_dates(start_date, end_date)
    
    # Check if the computed monthly dates match the number of price entries.
    if len(monthly_dates) != len(known_prices):
        print("Warning: The computed number of monthly dates does not match the number of price entries.")
        # Fallback: Use dates directly from the CSV.
        monthly_dates = [d for d in df['Dates'].dt.date.tolist()]
    
    # Compute days from the start_date for each monthly snapshot.
    days_from_start = np.array([(d - start_date).days for d in monthly_dates])
    
    # Fit the model to capture both the linear trend and seasonal variations.
    model_params = fit_trend_and_seasonality(days_from_start, known_prices)
    
    # Output the fitted model parameters.
    print("Fitted Model Parameters:")
    print(f"  Linear Trend: slope = {model_params['slope']:.4f}, intercept = {model_params['intercept']:.4f}")
    print(f"  Seasonal Component: amplitude = {model_params['amplitude']:.4f}, shift = {model_params['shift']:.4f}, period = {model_params['period']:.1f}")
    
    # Plot the original monthly prices and the fitted continuous model.
    plot_prices_and_fit(monthly_dates, known_prices, pd.Timestamp(start_date), model_params)
    
    # Interactive prompt: estimate price for a user-specified date.
    input_date_str = input("Enter a date (YYYY-MM-DD) for price estimation: ")
    try:
        query_date = pd.to_datetime(input_date_str)
    except Exception as e:
        print(f"Error parsing input date: {e}")
        sys.exit(1)
    
    try:
        estimated_price = interpolate_price(query_date, pd.Timestamp(start_date), monthly_dates, known_prices, model_params)
        print(f"Estimated natural gas price for {query_date.date()} is: {estimated_price:.2f}")
    except Exception as e:
        print(f"Error estimating price: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
