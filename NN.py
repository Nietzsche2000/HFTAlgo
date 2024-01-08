import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime


# ENSURE THAT YOU HAVE THE RIGHT VERSIONS OF THE LIBRARIES
# pip install pandas matplotlib torch

# DATA PREPROCESSING FUNCTION
def preprocess_data(df):
    # CONVERT 'DATE' TO A NUMERICAL FORMAT
    df['date'] = pd.to_datetime(df['date'])
    df['ordinal_date'] = df['date'].apply(lambda x: x.toordinal())

    # SELECT AND SPLIT THE DATA
    X = df[['ordinal_date', 'volume', 'elasticity', 'percent_change_7_days']]  # MODIFY WITH YOUR FEATURES
    y = df['price']

    # CONVERTING TO NUMPY ARRAY IF NOT ALREADY
    X = X.values if isinstance(X, pd.DataFrame) else X
    y = y.values if isinstance(y, pd.Series) else y

    # SPLIT THE DATA
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # SCALE THE FEATURES
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # CONVERT TO TORCH TENSORS
    X_train, X_test = torch.FloatTensor(X_train), torch.FloatTensor(X_test)
    y_train, y_test = torch.FloatTensor(y_train), torch.FloatTensor(y_test)

    return X_train, X_test, y_train, y_test


# NEURAL NETWORK MODEL
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 64)  # ADJUST THE INPUT DIMENSION AS NEEDED
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# TRAIN THE MODEL
def train(net, train_data, train_labels, EPOCHS=500, BATCH_SIZE=32):  # BATCH_SIZE set to 32
    trainloader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(train_data, train_labels),
        batch_size=BATCH_SIZE,
        shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(EPOCHS):  # LOOP OVER THE DATASET MULTIPLE TIMES
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            # ZERO THE PARAMETER GRADIENTS
            optimizer.zero_grad()

            # FORWARD + BACKWARD + OPTIMIZE
            outputs = net(inputs)
            loss = criterion(outputs, labels.view(-1, 1))
            loss.backward()
            optimizer.step()

        # PRINT STATISTICS EVERY 50 EPOCHS
        if epoch % 50 == 49:
            print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

    print('FINISHED TRAINING')


# PREDICT FUTURE PRICES
def predict(net, future_data):
    with torch.no_grad():
        future_data = torch.FloatTensor(future_data)
        predicted_prices = net(future_data)
    return predicted_prices.numpy()


def plot_predictions_with_training(train_dates, train_prices, future_dates, future_prices):
    plt.figure(figsize=(12, 6))

    # ASSUMING TRAIN_DATES AND FUTURE_DATES ARE ALREADY IN Pandas TIMESTAMP FORMAT
    # PLOTTING TRAINING DATA
    plt.plot(train_dates, train_prices, label='Training Prices', color='blue')

    # PLOTTING FUTURE PREDICTIONS
    plt.scatter(future_dates, future_prices, label='Future Predictions', color='red')

    plt.title('PRICE PREDICTIONS WITH TRAINING DATA')
    plt.xlabel('DATE')
    plt.ylabel('PRICE')
    plt.legend()
    plt.show()


# MAIN FUNCTION
def main():
    # LOAD AND PREPROCESS THE DATA
    df = pd.read_csv('730_market_data.csv')  # ENSURE YOU HAVE YOUR DATASET NAMED CORRECTLY
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # INITIALIZE THE MODEL
    net = Net()

    # TRAIN THE MODEL
    train(net, X_train, y_train)

    # EXTRACT TRAINING DATES AND PRICES FOR PLOTTING
    train_dates = df['date'][:len(y_train)]  # ASSUMING DATE IS ORDERED AND CORRESPONDS TO SPLIT
    train_prices = y_train.numpy()  # CONVERTING TRAINING LABELS TO NUMPY ARRAY FOR PLOTTING

    # PREPARE FUTURE DATA FOR PREDICTION (EXAMPLE FOR NEXT 5 DAYS)
    last_date = df['date'].max()
    last_volume = df['volume'].max()
    last_elasticity = df['elasticity'].max()
    last_percent_change_7_days = df['percent_change_7_days'].max()
    # future_dates = pd.date_range(start=last_date, periods=5, freq='D')
    # ordinal_dates = [d.toordinal() for d in future_dates]
    # CREATE FUTURE DATA ARRAY WITH ASSUMED CONSTANTS FOR OTHER FEATURES
    # future_data = np.array([[date, last_volume, last_elasticity, last_percent_change_7_days] for date in ordinal_dates])  # MODIFY BASED ON YOUR FEATURES
    # future_data = ordinal_dates  # MODIFY BASED ON YOUR FEATURES
    # print(future_data)
    # SCALE THE FUTURE DATA AS PER TRAINING
    # scaler = StandardScaler()
    # scaler.fit(X_train.numpy())  # FIT ON THE TRAINING DATA
    # future_data_scaled = scaler.transform(future_data)

    # PREDICT FUTURE PRICES
    future_prices = predict(net, X_train)

    # PLOT THE RESULTS WITH TRAINING DATA
    plot_predictions_with_training(train_dates, train_prices, train_dates, future_prices.flatten())


if __name__ == "__main__":
    main()
