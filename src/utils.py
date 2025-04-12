import pandas as pd
import matplotlib.pyplot as plt
import json

# Load model
def load_model_values(filename='../resources/model_values.json'):
    with open(filename, 'r') as f:
        model = json.load(f)
    return model['theta0'], model['theta1']

def load_data(filename):
    df = pd.read_csv(filename)
    return df['km'].values, df['price'].values

# Normalize and denormalize
def normalize(array):
    return (array - array.min()) / (array.max() - array.min())

def denormalize(value, original_array):
    return value * (original_array.max() - original_array.min()) + original_array.min()

def denormalize_thetas(mileage, price, theta0, theta1):
    price_range = price.max() - price.min()
    mileage_range = mileage.max() - mileage.min()

    real_theta1 = theta1 * (price_range / mileage_range)
    real_theta0 = theta0 * price_range + price.min() - real_theta1 * mileage.min()

    return real_theta0, real_theta1


def save_model(t0, t1, filename='../resources/model_values.json'):
    with open(filename, 'w') as f:
        json.dump({'theta0': t0, 'theta1': t1}, f)

# Plot bonus
def plot_regression(mileage_orig, price_orig, predicted_prices):
    plt.scatter(mileage_orig, price_orig, color='green', label='Actual prices')
    plt.plot(mileage_orig, predicted_prices, color='pink', label='Regression line')
    plt.xlabel('Mileage')
    plt.ylabel('Price')
    plt.title('Linear Regression: Price vs Mileage')
    plt.legend()
    plt.grid(False)
    plt.show()
