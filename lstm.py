import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn

import numpy as np
import torch.optim as optim
import torch.utils.data as data

df = pd.read_csv('2024-01-08_csgo_marketplace.csv')
timeseries = df[['price']].values.astype('float32')

# plt.plot(timeseries)
# plt.show()

# train-test split for time series
train_size = int(len(timeseries) * 0.80)
test_size = len(timeseries) - train_size
train, test = timeseries[:train_size], timeseries[train_size:]

def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset

    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset) - lookback):
        feature = dataset[i:i + lookback]
        target = dataset[i + 1:i + lookback + 1]
        X.append(feature)
        y.append(target)
    return torch.tensor(X), torch.tensor(y)

# def create_dataset_test(dataset, lookback, forward):
#     """Transform a time series into a prediction dataset
#
#     Args:
#         dataset: A numpy array of time series, first dimension is the time steps
#         lookback: Size of window for prediction
#     """
#     X, y = [], []
#     for i in range(len(dataset) - lookback):
#         feature = dataset[i:i + lookback]
#         target = dataset[i + 1:i + lookback + 1]
#         X.append(feature)
#         y.append(target)
#     return torch.tensor(X), torch.tensor(y)


lookback = 5
X_train, y_train = create_dataset(train, lookback)
X_test, y_test = create_dataset(test, lookback)
print(X_train.shape, y_train.shape)
# print(X_test.shape, y_test.shape)


class AirModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=3, batch_first=True)
        self.linear = nn.Linear(50, 50)
        self.linear2 = nn.Linear(50, 1)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        x = nn.functional.tanh(x)
        x = self.linear2(x)
        return x


model = AirModel()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)

n_epochs = 200
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    if epoch % 100 != 0:
        continue
    model.eval()
    with torch.no_grad():
        y_pred = model(X_train)
        train_rmse = np.sqrt(loss_fn(y_pred, y_train))
        y_pred = model(X_test)
        test_rmse = np.sqrt(loss_fn(y_pred, y_test))
    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))
    # print("Epoch %d: train RMSE %.4f" % (epoch, train_rmse))



with torch.no_grad():
    # shift train predictions for plotting
    train_plot = np.ones_like(timeseries) * np.nan
    y_pred = model(X_train)
    y_pred = y_pred[:, -1, :]
    train_plot[lookback:train_size] = model(X_train)[:, -1, :]
    # shift test predictions for plotting
    test_plot = np.ones_like(timeseries) * np.nan
    test_plot[train_size+lookback:len(timeseries)] = model(X_test)[:, -1, :]
# plot
plt.plot(timeseries, c='b')
plt.plot(train_plot, c='r')
plt.plot(test_plot, c='g')
plt.show()

# Recursive Prediction
# with torch.no_grad():
#     # Start with the last 'lookback' values from the training set
#     last_train_batch = X_train[-1].view(1, -1, 1)
#     future_predictions = []
#     train_plot = np.ones_like(timeseries) * np.nan
#     train_plot[lookback:train_size] = model(X_train)[:, -1, :]
#     for i in range(len(test)):
#         # Predict the next value
#         y_pred = model(last_train_batch)[:, -1, :]
#         future_predictions.append(y_pred.item())
#         # Update the input batch to include the new prediction
#         last_train_batch = torch.cat((last_train_batch[:, 1:, :], y_pred.view(1, 1, 1)), dim=1)
#
#     # Convert predictions list to numpy array
#     future_predictions = np.array(future_predictions)

# Plot
# plt.plot(timeseries, c='b', label='Original Series')
# plt.plot(np.arange(lookback, train_size), train_plot[lookback:train_size], c='r', label='Training Predictions')
# plt.plot(np.arange(train_size, len(timeseries)), future_predictions, c='g', label='Future Predictions')
# plt.legend()
# plt.show()