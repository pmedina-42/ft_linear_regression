
import numpy as np
import estimate
import pandas as pd
from utils import load_data, normalize, denormalize, denormalize_thetas, save_model, plot_regression


# Gradient descent
def train(mileage, price, learning_rate=0.042, iterations=2400):
    m = len(mileage)
    theta0 = 0.0
    theta1 = 0.0
    prev_mse = 0.0

    for i in range(iterations):
        predictions = estimate.estimate_price(mileage, theta0, theta1)
        errors = predictions - price

        tmp_theta0 = learning_rate * (1 / m) * errors.sum()
        tmp_theta1 = learning_rate * (1 / m) * (errors * mileage).sum()

        theta0 -= tmp_theta0
        theta1 -= tmp_theta1
        

    return theta0, theta1


# Main
if __name__ == "__main__":
    mileage, price = load_data('../resources/data.csv')

    mileage_norm = normalize(mileage)
    price_norm = normalize(price)

    normTheta0, normTheta1 = train(mileage_norm, price_norm)
    theta0, theta1 = denormalize_thetas(mileage, price, normTheta0, normTheta1)

    print(f"Training completed.\nθ0 = {theta0:.5f}\nθ1 = {theta1:.5f}")
    # Predict and denormalize
    data_predicted_price = estimate.estimate_price(mileage, theta0, theta1)
    predicted_price = denormalize(data_predicted_price, price)
    save_model(theta0, theta1)

    plot_regression(mileage, price, data_predicted_price)
