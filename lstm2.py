import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.utils.data as data

df = pd.read_csv('2024-01-08_csgo_marketplace.csv')
timeseries = df[['price']].values.astype('float32')

# train-test split for time series
train_size = int(len(timeseries) * 0.80)
test_size = len(timeseries) - train_size
train, test = timeseries[:train_size], timeseries[train_size:]


def create_dataset(dataset, lookback, prediction_days):
    X, y = [], []
    for i in range(len(dataset) - lookback - prediction_days):
        feature = dataset[i:i + lookback]
        target = dataset[i + lookback + prediction_days - 1]
        X.append(feature)
        y.append(target)
    # Convert lists to NumPy arrays before converting to tensors
    X, y = np.array(X), np.array(y)
    return torch.tensor(X), torch.tensor(y)



lookback = 5
prediction_days = 100  # Predict 100 days ahead
X_train, y_train = create_dataset(train, lookback, prediction_days)
X_test, y_test = create_dataset(test, lookback, prediction_days)

class AirModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, prediction_days)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x


model = AirModel()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)

n_epochs = 200
# Adjust model training to match target tensor shape
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        y_pred = y_pred[:, -1, 0]
        loss = loss_fn(y_pred.unsqueeze(-1), y_batch)  # Adjust prediction tensor shape
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    if epoch % 100 != 0:
        continue
    model.eval()
    with torch.no_grad():
        y_pred_train = model(X_train)[:, -1, 0]
        train_rmse = np.sqrt(loss_fn(y_pred_train.unsqueeze(-1), y_train))
        y_pred_test = model(X_test)[:, -1, 0]
        test_rmse = np.sqrt(loss_fn(y_pred_test.unsqueeze(-1), y_test))
    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse.item(), test_rmse.item()))

# Prepare train and test predictions for plotting
with torch.no_grad():
    train_plot = np.empty(len(timeseries))
    train_plot[:] = np.nan
    y_pred_train = model(X_train)[:, -1, 0]

    # Adjust the range for train_plot to match the size of y_pred_train
    train_plot[lookback + prediction_days - 1:lookback + prediction_days - 1 + len(y_pred_train)] = y_pred_train.numpy()

    test_plot = np.empty(len(timeseries))
    test_plot[:] = np.nan
    y_pred_test = model(X_test)[:, -1, 0]

    # Adjust the range for test_plot to match the size of y_pred_test
    test_plot[train_size + lookback + prediction_days - 1:train_size + lookback + prediction_days - 1 + len(
        y_pred_test)] = y_pred_test.numpy()

# plot
plt.figure(figsize=(15, 5))
plt.plot(timeseries, label='Original Series')
plt.plot(train_plot, label='Train Predictions', alpha=0.7)
plt.plot(test_plot, label='Test Predictions', alpha=0.7)
plt.legend()
plt.show()

