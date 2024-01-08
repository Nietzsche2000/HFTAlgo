import requests
import json
import pandas as pd
from datetime import datetime
from datetime import timedelta

def preprocess_and_merge_csgo_tables(df1, df2):
    """
    Preprocesses and merges two dataframes based on the year and month.
    The first dataframe is expected to have a 'Month' column with some non-standard date formats.
    The second dataframe is expected to have a 'date' column in a standard format like '2018-12-06'.
    """
    # Remove non-standard date formats from 'Month' column
    df1 = df1[df1['Month'].str.contains(r'\b\d{4}\b')]  # Keeping rows with a four-digit year

    # Convert 'Month' column to datetime format
    df1['YearMonth'] = pd.to_datetime(df1['Month']).dt.to_period('M')

    # Convert 'date' column to datetime format and extract year-month
    df2['YearMonth'] = pd.to_datetime(df2['date']).dt.to_period('M')

    # Merging the two dataframes based on the YearMonth column
    merged_df = pd.merge(df1, df2, on='YearMonth', how='outer')

    return merged_df

# Applying the modified merge function
merged_df = preprocess_and_merge_csgo_tables(df1, df2)