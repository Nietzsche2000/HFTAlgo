## Table of Contents
1. [Project Description](#project-description)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Credits](#credits)
6. [License](#license)

## Description
This Python script is dedicated to fetching, processing, and analyzing market data for Counter-Strike: Global Offensive (CS:GO) items on Steam. It includes functionalities like calculating price elasticity, percent changes, merging datasets, and updating data with U.S. holiday information.

## Features
- **Historical Data Collection**: Fetches data from Steam's market.
- **Elasticity and Percent Change Calculations**: Computes various financial metrics.
- **Dataframe Updates**: Enhances dataframes with new metrics and holiday information.
- **Data Merging**: Combines data from multiple CS:GO sources for detailed analysis.

## Installation
This script requires Python 3.x and the following libraries: `requests`, `json`, `pandas`, `datetime`, and `pandas.tseries.holiday`. Install them using pip:
```bash
pip install requests pandas
```

## Usage
To run the script:
1. Update the `cookie` and `game_id` in the `main()` function with your credentials.
2. Execute the script in a Python environment.
3. Output is in the same folder as 2 csv files.
   
date_csgo_marketplace -> just marketplace data

date_combined_csgo_data -> marketplace data and csgo game play data

Example:
```python
cookie = {'steamLoginSecure': 'your_cookie_here'}
game_id = '730'
item_name = 'AWP | Neo-Noir (Factory New)'
main()
```

## Credits
- **Author**: [Ryan Soohoo, Monishwaren Maheswaran]
- Information Used: https://www.blakeporterneuro.com/learning-python-project-3-scrapping-data-from-steams-community-market/

## License
This project is licensed under the [MIT License](LICENSE.md). A copy of the license is provided in the repository.

---
