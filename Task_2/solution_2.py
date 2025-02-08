#!/usr/bin/env python
"""
Natural Gas Contract Pricing Model

This script calculates the value of a natural gas storage/trading contract based on:
    - The purchase price (buy price) of natural gas.
    - The sale price (sell price) of natural gas.
    - The volume of natural gas traded (in MMBtu).
    - The storage fee per month.
    - The storage duration in months.
    - The injection/withdrawal fee per million MMBtu.
    - The transportation fee per trip (typically for both injection and withdrawal).
    - The number of transportation trips.

The contract value is computed as:

    Contract Value = (Sell Price - Buy Price) * Volume 
                     - (Storage Fee * Storage Duration)
                     - (Injection/Withdrawal Fee * (Volume / 1e6))
                     - (Transportation Fee * Number of Trips)

Usage (from the command line):
    python pricing_model.py --buy_price 2 --sell_price 3 --volume 1000000 \
        --storage_fee 100000 --storage_duration 4 --injection_fee 10000 \
        --transport_fee 50000 --num_transports 2
"""

import argparse
import sys
import matplotlib.pyplot as plt

def calculate_contract_value(buy_price: float,
                             sell_price: float,
                             volume: float,
                             storage_fee: float,
                             storage_duration: int,
                             injection_withdrawal_fee: float,
                             transportation_fee: float,
                             num_transports: int = 2) -> dict:
    """
    Calculate the contract value and cost breakdown for a natural gas trading contract.

    Parameters:
        buy_price (float): Purchase price per MMBtu ($/MMBtu).
        sell_price (float): Sale price per MMBtu ($/MMBtu).
        volume (float): Volume of natural gas traded in MMBtu.
        storage_fee (float): Storage fee per month in dollars.
        storage_duration (int): Duration of storage in months.
        injection_withdrawal_fee (float): Injection/Withdrawal fee per 1e6 MMBtu.
        transportation_fee (float): Transportation fee per trip in dollars.
        num_transports (int): Number of transportation trips (default is 2).

    Returns:
        dict: Breakdown of the contract value, including revenue, individual cost components,
              and the final contract value.
    """
    # Calculate revenue: profit per unit times volume
    revenue = (sell_price - buy_price) * volume

    # Calculate total storage cost (storage fee per month * number of months)
    total_storage_cost = storage_fee * storage_duration

    # Calculate injection/withdrawal cost (fee scaled by volume in million MMBtu)
    total_injection_withdrawal_cost = injection_withdrawal_fee * (volume / 1e6)

    # Calculate transportation cost (fee per trip * number of trips)
    total_transportation_cost = transportation_fee * num_transports

    # The final contract value is the net profit after deducting all costs
    contract_value = revenue - total_storage_cost - total_injection_withdrawal_cost - total_transportation_cost

    return {
        'revenue': revenue,
        'storage_cost': total_storage_cost,
        'injection_withdrawal_cost': total_injection_withdrawal_cost,
        'transportation_cost': total_transportation_cost,
        'contract_value': contract_value
    }

def plot_cost_breakdown(breakdown: dict):
    """
    Plot a bar chart showing the revenue, individual costs, and final contract value.

    Parameters:
        breakdown (dict): Dictionary containing 'revenue', cost items, and 'contract_value'.
    """
    labels = ['Revenue', 'Storage Cost', 'Inj/WD Cost', 'Transport Cost', 'Final Value']
    # For visualization, we display costs as negative (since they reduce the revenue)
    values = [
        breakdown['revenue'],
        -breakdown['storage_cost'],
        -breakdown['injection_withdrawal_cost'],
        -breakdown['transportation_cost'],
        breakdown['contract_value']
    ]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color=['green', 'red', 'red', 'red', 'blue'])
    plt.title('Natural Gas Contract Pricing Breakdown')
    plt.ylabel('Dollars ($)')
    
    # Annotate bars with their respective values
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'${yval:,.0f}', ha='center', va='bottom')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def parse_arguments():
    """
    Parse command-line arguments for the pricing model.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Natural Gas Contract Pricing Model")
    parser.add_argument('--buy_price', type=float, default=2.0, help="Buy price per MMBtu ($)")
    parser.add_argument('--sell_price', type=float, default=3.0, help="Sell price per MMBtu ($)")
    parser.add_argument('--volume', type=float, default=1e6, help="Volume in MMBtu (default: 1e6)")
    parser.add_argument('--storage_fee', type=float, default=100000.0, help="Storage fee per month ($)")
    parser.add_argument('--storage_duration', type=int, default=4, help="Storage duration in months")
    parser.add_argument('--injection_fee', type=float, default=10000.0, help="Injection/Withdrawal fee per 1e6 MMBtu ($)")
    parser.add_argument('--transport_fee', type=float, default=50000.0, help="Transportation fee per trip ($)")
    parser.add_argument('--num_transports', type=int, default=2, help="Number of transportation trips (default: 2)")
    return parser.parse_args()

def main():
    """
    Main function to calculate and display the contract pricing.
    """
    args = parse_arguments()

    # Calculate the contract value using provided or default parameters
    try:
        breakdown = calculate_contract_value(
            buy_price=args.buy_price,
            sell_price=args.sell_price,
            volume=args.volume,
            storage_fee=args.storage_fee,
            storage_duration=args.storage_duration,
            injection_withdrawal_fee=args.injection_fee,
            transportation_fee=args.transport_fee,
            num_transports=args.num_transports
        )
    except Exception as e:
        print(f"Error calculating contract value: {e}")
        sys.exit(1)

    # Display the cost breakdown
    print("\nContract Pricing Breakdown:")
    print(f"Revenue:                    ${breakdown['revenue']:,.2f}")
    print(f"Storage Cost:              -${breakdown['storage_cost']:,.2f}")
    print(f"Injection/Withdrawal Cost: -${breakdown['injection_withdrawal_cost']:,.2f}")
    print(f"Transportation Cost:       -${breakdown['transportation_cost']:,.2f}")
    print(f"Final Contract Value:       ${breakdown['contract_value']:,.2f}")

    # Generate a bar chart for visual comparison
    plot_cost_breakdown(breakdown)

if __name__ == "__main__":
    main()
