import requests
import json
import pandas as pd
from datetime import datetime, timedelta
from pandas.tseries.holiday import USFederalHolidayCalendar
import os

def calculate_elasticity(df, start_date, end_date):
    data_window = df[(df['date'] <= end_date) & (df['date'] > start_date)]
    if not data_window.empty:
        avg_price = data_window['price'].mean()
        avg_volume = data_window['volume'].mean()
        return avg_price, avg_volume
    return None, None

def calculate_percent_change(current_price, comparison_price):
    if comparison_price != 0:
        return ((current_price - comparison_price) / comparison_price) * 100
    return 0

def update_dataframe(df, holidays):
    for index, row in df.iterrows():
        # Elasticity
        start_date = row['date'] - timedelta(days=60)
        avg_price, avg_volume = calculate_elasticity(df, start_date, row['date'])
        if avg_price is not None and avg_volume is not None:
            percentage_change_price = calculate_percent_change(row['price'], avg_price)
            percentage_change_volume = calculate_percent_change(row['volume'], avg_volume)
            df.at[index, 'elasticity'] = percentage_change_volume / percentage_change_price if percentage_change_price != 0 else 0
        
        # Percent Change 7 Days
        start_date_7_days = max(row['date'] - timedelta(days=7), df['date'].min())
        data_comparison_date = df[df['date'] <= start_date_7_days].tail(1)
        if not data_comparison_date.empty:
            df.at[index, 'percent_change_7_days'] = calculate_percent_change(row['price'], data_comparison_date.iloc[0]['price'])

        # Percent Change 1 Day
        previous_day = row['date'] - timedelta(days=1)
        data_previous_day = df[df['date'] == previous_day]
        if not data_previous_day.empty:
            df.at[index, 'percent_change_1_day'] = calculate_percent_change(row['price'], data_previous_day.iloc[0]['price'])

        # US Holiday
        df.at[index, 'US_Holiday'] = row['date'] in holidays


def fetch_price_history(game_id, item_name, cookie):
    item_name_encoded = requests.utils.quote(item_name)
    url = f'https://steamcommunity.com/market/pricehistory/?appid={game_id}&market_hash_name={item_name_encoded}'
    response = requests.get(url, cookies=cookie)
    if response.status_code != 200:
        return None

    data = json.loads(response.content)
    if not data or 'prices' not in data:
        return None

    price_history = [{'date': datetime.strptime(entry[0][0:11], '%b %d %Y'),
                      'price': float(entry[1]), 
                      'volume': int(entry[2])} for entry in data['prices']]
    
    df = pd.DataFrame(price_history)
    df.sort_values(by='date', inplace=True)
    df['elasticity'] = 0.0
    df['percent_change_7_days'] = 0.0
    df['percent_change_1_day'] = 0.0
    df['US_Holiday'] = False

    start_year, end_year = df['date'].min().year, df['date'].max().year
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=f'{start_year}-01-01', end=f'{end_year}-12-31')

    update_dataframe(df, holidays)

    return df

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

# Main function and other parts of the script remain unchanged.

def main():
    
    cookie = {'steamLoginSecure': '76561198285682461%7C%7CeyAidHlwIjogIkpXVCIsICJhbGciOiAiRWREU0EiIH0.eyAiaXNzIjogInI6MERENF8yMzkwMURBQ19GMUMwOCIsICJzdWIiOiAiNzY1NjExOTgyODU2ODI0NjEiLCAiYXVkIjogWyAid2ViIiBdLCAiZXhwIjogMTcwNDc5NTk1MCwgIm5iZiI6IDE2OTYwNjkxMzcsICJpYXQiOiAxNzA0NzA5MTM3LCAianRpIjogIjBERDBfMjNBQkNFQjRfQTFENTAiLCAib2F0IjogMTcwMTMwODIxNSwgInJ0X2V4cCI6IDE3MTkzNDk1MDksICJwZXIiOiAwLCAiaXBfc3ViamVjdCI6ICIxMzUuMTgwLjE5OS4yMzkiLCAiaXBfY29uZmlybWVyIjogIjEzNS4xODAuMTk5LjIzOSIgfQ.9vVEWTygSc6CWsgj5TPQvTCJL4fxcOlF3xDbOvnKobZI_b2FWGw7-Co0gx_2im33h7ofw25Qsb0fp0GzXYUbBg'}
    game_id = '730'  # Example: Counter-Strike: Global Offensive

    all_items = ['AWP | Neo-Noir (Factory New)']
    all_data = []

    for item_name in all_items:
        print(f"Fetching data for: {item_name}")
        price_data = fetch_price_history(game_id, item_name, cookie)
        if price_data is not None:
            price_data['item_name'] = item_name
            all_data.append(price_data)

    # Fetch and preprocess the CS:GO data
    full_data = pd.concat(all_data)

   # Save the data in the same directory as the script
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_file = os.path.abspath(os.path.dirname(__file__)) #older/folder2/scripts_folder

    marketplace_filename = f"{current_date}_csgo_marketplace.csv"
    marketplace_full_path = os.path.join(current_file, marketplace_filename)
    full_data.to_csv(marketplace_full_path, index=False)
    print(f"Data collection complete. CSV file created at {marketplace_full_path}")

    # Load the second CSV file
    month_play_filename = os.path.join(current_file, 'csgo_month_play (1).csv')
    month_play_data = pd.read_csv(month_play_filename)

    # Preprocess and merge the tables
    combined_data = preprocess_and_merge_csgo_tables(month_play_data, full_data)

    # Save the combined data to a CSV file in the same directory
    combined_filename = f"{current_date}_combined_csgo_data.csv"
    combined_full_path = os.path.join(current_file, combined_filename)
    combined_data.to_csv(combined_full_path, index=False)
    print(f"Combined data saved to {combined_full_path}")

if __name__ == "__main__":
    main()