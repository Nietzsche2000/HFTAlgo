import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# DATA PREPROCESSING
def preprocess_data(df):
    # CONVERT 'DATE' TO DATETIME
    df['date'] = pd.to_datetime(df['date'])

    # SPLIT DATA INTO FEATURES AND TARGET
    X = df.drop(columns=['price'])
    y = df['price']

    # DEFINE CATEGORICAL AND NUMERICAL FEATURES
    categorical_features = ['day_of_week', 'item_name']
    numerical_features = ['volume', 'elasticity', 'percent_change_7_days']

    # CREATE TRANSFORMERS FOR NUMERICAL AND CATEGORICAL DATA
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # COMBINE TRANSFORMERS INTO A PREPROCESSOR WITH COLUMNTRANSFORMER
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor, X, y


# MODEL TRAINING
def train_model(X, y, preprocessor):
    # SPLIT DATA INTO TRAINING AND TESTING SETS
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # CREATE A PIPELINE WITH PREPROCESSING AND MODEL
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', RandomForestRegressor(random_state=42))])

    # TRAIN THE MODEL
    model.fit(X_train, y_train)

    # PREDICT AND EVALUATE THE MODEL
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'MEAN SQUARED ERROR: {mse}')

    return model


# PREDICTION
def predict_prices(model, future_data):
    # CONVERT FUTURE DATA TO DATAFRAME
    future_df = pd.DataFrame(future_data)

    # PREDICT THE PRICE FOR THE GIVEN FUTURE DATA
    predicted_prices = model.predict(future_df)
    return predicted_prices


# PLOTTING FUNCTION
def plot_predictions(df, model, future_dates, actuals=True):
    # EXTRACT TRAINING DATA FOR PLOTTING
    dates = df['date']
    prices = df['price']

    # PREPARE FUTURE DATA FOR PREDICTION AND PLOTTING
    future_data = [{'date': date, 'day_of_week': date.strftime('%A'), 'volume': 1000,
                    'elasticity': 0.5, 'percent_change_7_days': 0.03, 'item_name': 'Item1'}
                   for date in future_dates]
    future_df = pd.DataFrame(future_data)
    future_prices = predict_prices(model, future_df)

    # PLOTTING
    plt.figure(figsize=(12, 6))
    if actuals:
        plt.plot(dates, prices, label='Historical Prices', color='blue')
    plt.plot(future_dates, future_prices, label='Future Predictions', color='red', linestyle='--')
    plt.title('Price Predictions')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


# MAIN FUNCTION
def main():
    # LOAD YOUR DATASET
    df = pd.read_csv('730_market_data.csv')
    df['date'] = pd.to_datetime(df['date'])

    # PREPROCESS AND TRAIN MODEL
    preprocessor, X, y = preprocess_data(df)
    model = train_model(X, y, preprocessor)

    # DEFINE FUTURE DATES FOR PREDICTION
    last_date = df['date'].min()
    future_dates = pd.date_range(start=last_date, periods=1000, freq='D')  # EDIT AS NEEDED

    # PLOT HISTORICAL DATA AND FUTURE PROJECTIONS
    plot_predictions(df, model, future_dates)


if __name__ == "__main__":
    main()