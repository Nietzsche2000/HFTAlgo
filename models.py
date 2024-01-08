from sklearn.linear_model import LinearRegression
import Data

class Model:

    def __init__(self, data: Data.Data):
        self.data = data


class LinReg(Model):

    def __init__(self):
        self.model = LinearRegression()

    def train(self, x, y):
        self.model.fit(self.data.get_training_data()[x], self.data.get_training_data()[y])

    def predict(self):
        self.model.predict(self.data.get_prediction_data())

    def test(self):
        y_preds = self.model.predict(self.data.get_test_data())
        return self.data.compare_with_test(y_preds)

