import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# DEVICE CONFIGURATION
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DataPreprocessor:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)

    def preprocess(self):
        # CONVERT DATE TO ORDINAL
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data['date'] = self.data['date'].map(pd.Timestamp.toordinal)

        # ASSUMING 'item_name' IS THE COLUMN CAUSING ISSUES
        # EXCLUDING 'item_name' FROM DATA
        # self.data = self.data.drop(columns=['item_name'])

        # NORMALIZE THE FEATURES
        scaler = MinMaxScaler(feature_range=(-1, 1))
        features_to_scale = ['date', 'price', 'volume', 'elasticity', 'percent_change_7_days']
        self.data[features_to_scale] = scaler.fit_transform(self.data[features_to_scale])
        return self.data, scaler


# LSTM MODEL
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq.view(len(input_seq), 1, -1))
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


# TRAINING THE MODEL
def train_model(model, train_loader, num_epochs):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(num_epochs):
        for seq, labels in train_loader:
            optimizer.zero_grad()
            y_pred = model(seq)
            single_loss = criterion(y_pred, labels)
            single_loss.backward()
            optimizer.step()
        if epoch % 25 == 1:
            print(f'Epoch {epoch} Loss: {single_loss.item()}')


# MAIN SCRIPT
if __name__ == "__main__":
    # DATA PREPARATION
    data_preprocessor = DataPreprocessor('730_market_data.csv')
    data, scaler = data_preprocessor.preprocess()
    X = data[['date', 'volume', 'elasticity', 'percent_change_7_days']].values
    y = data['price'].values
    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).to(device)
    train_data = TensorDataset(X, y)
    train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)

    # MODEL INITIALIZATION
    model = LSTM(input_size=5, hidden_layer_size=100, num_layers=1, output_size=1).to(device)
    train_model(model, train_loader, num_epochs=100)

    # PREDICTIONS AND PLOTTING
    model.eval()
    predictions = []
    with torch.no_grad():
        for seq in X:
            y_pred = model(seq.view(1, -1))
            predictions.append(y_pred.cpu().numpy())
    predictions = np.array(predictions)
    actual_prices = scaler.inverse_transform(data[['price']])
    predicted_prices = scaler.inverse_transform(np.concatenate((data.drop('price', axis=1), predictions), axis=1))[:,
                       -1]

    plt.figure(figsize=(15, 7))
    plt.plot(data['date'], actual_prices, label='Actual Price')
    plt.plot(data['date'], predicted_prices, label='Predicted Price', color='red')
    plt.title('Date vs Price: Actual and Predicted')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(visible=True)
    plt.show()
