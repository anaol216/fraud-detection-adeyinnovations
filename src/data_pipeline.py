# src/data_pipeline.py

import pandas as pd
import numpy as np
import os

# === CONFIG ===
RAW_DATA_PATH = 'data/Fraud_Data.csv'
IP_COUNTRY_PATH = 'data/IpAddress_to_Country.csv'
OUTPUT_PATH = 'outputs/processed_data/merged_fraud_data.csv'

# === FUNCTIONS ===

def ip_to_country(ip_int, ip_df):
    """
    Map a numeric IP to its corresponding country using IP range lookup.
    """
    row = ip_df[
        (ip_df['lower_bound_ip_address'] <= ip_int) &
        (ip_df['upper_bound_ip_address'] >= ip_int)
    ]
    return row['country'].values[0] if not row.empty else 'Unknown'

def load_and_merge_data():
    """
    Load Fraud_Data.csv and IpAddress_to_Country.csv,
    map each IP address to country, and save merged dataset.
    """
    print(" Loading datasets...")
    df = pd.read_csv(RAW_DATA_PATH)
    ip_df = pd.read_csv(IP_COUNTRY_PATH)

    # Ensure IP bounds are integers
    ip_df['lower_bound_ip_address'] = ip_df['lower_bound_ip_address'].astype(int)
    ip_df['upper_bound_ip_address'] = ip_df['upper_bound_ip_address'].astype(int)

    print(" Converting IP addresses...")
    # Convert ip_address (possibly float) to int directly
    df['ip_int'] = df['ip_address'].astype(float).astype(int)

    print(" Mapping IPs to countries...")
    df['country'] = df['ip_int'].apply(lambda x: ip_to_country(x, ip_df))

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f" Merged data saved to {OUTPUT_PATH}")

# === RUN ===

if __name__ == '__main__':
    load_and_merge_data()
