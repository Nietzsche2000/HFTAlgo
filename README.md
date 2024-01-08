
# ReadMe for GitHub Project: Steam Market Data Analysis Script

## Project Title
Steam Market Data Analysis Script

## Description
This Python script is dedicated to fetching, processing, and analyzing market data for Counter-Strike: Global Offensive (CS:GO) items on Steam. It includes functionalities like calculating price elasticity, percent changes, merging datasets, and updating data with U.S. holiday information.

## Table of Contents
1. [Project Description](#project-description)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Contributing](#contributing)
6. [Credits](#credits)
7. [License](#license)

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
Example:
```python
cookie = {'steamLoginSecure': 'your_cookie_here'}
game_id = '730'
item_name = 'AWP | Neo-Noir (Factory New)'
main()
```

## Contributing
Contributions to improve the script are welcome. Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## Credits
- **Author**: [Ryan Soohoo, Monishwaren Maheswaran]

## License
This project is licensed under the [MIT License](LICENSE.md). A copy of the license is provided in the repository.

---

**Note**: Keep your README updated. This file is a starting point and should evolve with your project. Use clear and concise language to make it accessible to a wide audience. Happy Coding!

*For more information or to report issues, please visit the [GitHub Repository](Your GitHub Repository Link).*
