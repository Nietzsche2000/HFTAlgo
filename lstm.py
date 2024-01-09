import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def preprocess_data(data, feature_cols, target_col):
    # Normalizing the dataset
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(data[feature_cols])

    # Creating sequences for LSTM
    X, y = [], []
    for i in range(len(scaled_data) - 1):
        X.append(scaled_data[i, :])
        y.append(scaled_data[i + 1, target_col])

    return np.array(X), np.array(y)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


def train_model(model, train_data, learning_rate, epochs):
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for seq, labels in train_data:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                 torch.zeros(1, 1, model.hidden_layer_size))
            y_pred = model(seq)
            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

        if epoch % 25 == 0:
            print(f'epoch {epoch} loss: {single_loss.item()}')
    return model


def predict(model, input_data, feature_cols):
    model.eval()
    with torch.no_grad():
        input_data = torch.FloatTensor(input_data).view(-1, 1, len(feature_cols))
        prediction = model(input_data)
        return prediction.item()


def plot_predictions(actual, predicted):
    plt.figure(figsize=(10,6))
    plt.plot(actual, label='Actual Price')
    plt.plot(predicted, label='Predicted Price')
    plt.title('Price Prediction on Training Data')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


def main():
    # Load and preprocess data
    data = pd.read_csv('2024-01-08_csgo_marketplace.csv')
    feature_cols = ['price', 'volume', 'elasticity', 'percent_change_7_days', 'percent_change_1_day']
    target_col = data.columns.get_loc('price')
    X, y = preprocess_data(data, feature_cols, target_col)

    # Define model parameters
    input_size = len(feature_cols)
    hidden_layer_size = 50
    output_size = 1
    learning_rate = 0.001
    epochs = 150

    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)

    # Create LSTM model
    model = LSTMModel(input_size, hidden_layer_size, output_size)

    # Train the model
    train_data = DataLoader([(X_tensor[i], y_tensor[i]) for i in range(len(X_tensor))], batch_size=1, shuffle=False)
    model = train_model(model, train_data, learning_rate, epochs)

    # Make predictions on the training data
    model.eval()
    predicted_prices = []
    with torch.no_grad():
        for i in range(len(X_tensor)):
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                 torch.zeros(1, 1, model.hidden_layer_size))
            predicted_prices.append(model(X_tensor[i].view(-1, 1, input_size)).item())

    # Plot predictions against actual data
    actual_prices = data['price'][
                    1:].values  # Excluding the first value as we start predicting from the second data point
    plot_predictions(actual_prices, predicted_prices)


if __name__ == "__main__":
    main()
