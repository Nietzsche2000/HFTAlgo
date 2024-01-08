import pandas as pd


class Data:

    def __init__(self, csv: str):
        self.data = pd.read_csv(csv)

    def tensor_train(self):
        return

    def compress(self, factor=1):
        return

    def get_training_data(self):
        return self.data['X'], self.data['Y']

    def get_test_data(self):
        return self.data

    def get_prediction_data(self):
        return self.data

    def compare_with_test(self, y_preds):
        return
