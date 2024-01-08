import requests
import json
import pandas as pd
from datetime import datetime

def get_all_item_names(game_id, cookie):
    item_names = []
    start = 0
    count = 100
    total_items = 10
    iteration = 0  # New variable to count iterations
    number_of_iterations = 10
    while total_items is None or start < total_items:
        if iteration >= number_of_iterations:  # Check if 10 iterations have been done
            break

        url = f'https://steamcommunity.com/market/search/render/?search_descriptions=0&sort_column=default&sort_dir=desc&appid={game_id}&norender=1&count={count}&start={start}'
        response = requests.get(url, cookies=cookie)
        if response.status_code != 200:
            break

        data = json.loads(response.content)
        if total_items is None:
            total_items = data['total_count']

        for item in data['results']:
            item_names.append(item['hash_name'])
        print("getting item name:" + item['hash_name'])
        start += count
        iteration += 1  # Increment the iteration counter

    return item_names

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
                      'volume': int(entry[2])} 
                     for entry in data['prices']]
    print("current: " + item_name);
    return pd.DataFrame(price_history)

def main():
    cookie = {'steamLoginSecure': '76561198285682461%7C%7CeyAidHlwIjogIkpXVCIsICJhbGciOiAiRWREU0EiIH0.eyAiaXNzIjogInI6MERENF8yMzkwMURBQ19GMUMwOCIsICJzdWIiOiAiNzY1NjExOTgyODU2ODI0NjEiLCAiYXVkIjogWyAid2ViIiBdLCAiZXhwIjogMTcwNDc5NTk1MCwgIm5iZiI6IDE2OTYwNjkxMzcsICJpYXQiOiAxNzA0NzA5MTM3LCAianRpIjogIjBERDBfMjNBQkNFQjRfQTFENTAiLCAib2F0IjogMTcwMTMwODIxNSwgInJ0X2V4cCI6IDE3MTkzNDk1MDksICJwZXIiOiAwLCAiaXBfc3ViamVjdCI6ICIxMzUuMTgwLjE5OS4yMzkiLCAiaXBfY29uZmlybWVyIjogIjEzNS4xODAuMTk5LjIzOSIgfQ.9vVEWTygSc6CWsgj5TPQvTCJL4fxcOlF3xDbOvnKobZI_b2FWGw7-Co0gx_2im33h7ofw25Qsb0fp0GzXYUbBg'}  # Replace with your actual cookie
    game_id = '730'  # Example: Counter-Strike: Global Offensive

    all_items = get_all_item_names(game_id, cookie)
    all_data = []

    for item_name in all_items:
        print(f"Fetching data for: {item_name}")
        price_data = fetch_price_history(game_id, item_name, cookie)
        if price_data is not None:
            price_data['item_name'] = item_name
            all_data.append(price_data)

    full_data = pd.concat(all_data)
    full_data.to_csv(f'{game_id}_market_data.csv', index=False)
    print("Data collection complete. CSV file created.")
    current_date = datetime.now().strftime("%Y-%m-%d")
    filename = f"{current_date}_csgo_marketplace.csv"
    full_path = f"C:\\Users\\ryans\\{filename}"

    full_data.to_csv(full_path, index=False)
    print(f"Data collection complete. CSV file created at {full_path}")

if __name__ == "__main__":
    main()
