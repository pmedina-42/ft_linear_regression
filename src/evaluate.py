from utils import load_model_values, load_data
from estimate import estimate_price
import numpy as np

mileage, price = load_data('../resources/data.csv')
t0, t1 = load_model_values()
predictions = estimate_price(mileage, t0, t1)


# MSE and RMSE tell us how far off our model's predictions are from the real prices.
# They care a lot about big mistakes — if the model gets a few prices *really* wrong,
# that will hurt the score a lot.
# RMSE is easier to understand than MSE because it's in the same units as price
# (like euros or dollars), so it tells us, on average, how many units off we are.

# MAE is another way to check how off the predictions are, but it’s simpler.
# It just adds up how wrong we were (without squaring) and gives the average error.
# It’s better if you don’t want big mistakes to count *extra* like with MSE.

# R² (read: “R squared”) shows how well the model explains the real prices.
# It goes from 0 to 1, and more is better:
#   R² = 1   → the model is perfect
#   R² = 0   → the model is useless 
#   R² < 0   → you might as well go to a pythoness because the model is worse than a dumb guess
def evaluate():
    errors = predictions - price
    mean_squared_error = np.mean(errors ** 2)
    root_mean_squared_error = np.sqrt(mean_squared_error)
    mean_absolute_error = np.mean(np.abs(errors))
    
    unexplained_variance = np.sum(errors ** 2)
    total_variance = np.sum((price - np.mean(price)) ** 2)
    r2 = 1 - (unexplained_variance / total_variance)

    print("Model Evaluation:\n")
    print(f"- MSE  (Mean Squared Error): {mean_squared_error:.3f}")
    print(f"- RMSE (Root Mean Squared Error): {root_mean_squared_error:.3f}")
    print(f"- MAE  (Mean Absolute Error): {mean_absolute_error:.3f}")
    print(f"- R²   (R-squared): {r2:.5f}")

if __name__=="__main__":
    try:
        evaluate()
    except Exception as e:
        print('Something went wrong')
