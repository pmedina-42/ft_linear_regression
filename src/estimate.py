from utils import load_model_values

# Hypothesis function
def estimate_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)   

if __name__ == "__main__":
    try:
        t0, t1 = load_model_values()

        mileage_input = float(input("Enter the mileage: "))
        estimated_price = estimate_price(mileage_input, t0, t1)
        print(f"Estimated price: {estimated_price:.2f} €")
    except Exception as e:
        print('Something went wrong')
