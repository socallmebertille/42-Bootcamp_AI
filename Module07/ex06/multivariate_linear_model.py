import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ex05.mylinearregression import MyLinearRegression as MyLR

def train_and_evaluate_model(X, Y, thetas_init, alpha, max_iter, feature_name, model_name):
    model = MyLR(thetas=thetas_init, alpha=alpha, max_iter=max_iter)
    model.fit_(X, Y)
    y_hat = model.predict_(X)
    mse = model.mse(Y, y_hat)
    model.plot_data(X, Y, False, model_name, feature_name, 'Sell price (in keuros)')
    return model, mse

def train_multivariate_model(X, Y, feature_names, feature_des, model_name):
    model =  MyLR(thetas=[1.0, 1.0, 1.0, 1.0], alpha=9e-5, max_iter=500000)
    model.fit_(X, Y)
    y_hat = model.predict_(X)
    mse = model.mse(Y, y_hat)
    y_color = ["midnightblue", "mediumseagreen", "darkviolet"]
    y_hat_color = ["cornflowerblue", "chartreuse", "violet"]
    for i, feature in enumerate(feature_names):
        plt.figure()
        plt.scatter(X[:, i], Y, label="sell price", color=y_color[i])
        plt.scatter(X[:, i], y_hat, label="predicted sell price", color=y_hat_color[i])
        plt.xlabel(feature_des[i])
        plt.ylabel("y: sell price (in keuros)")
        plt.legend()
        plt.title(f"Actual vs Predicted prices w.r.t {feature}") # with respect to
        plt.show()

    return model, mse

def main():
    """Univariate Linear Regression
    Train three different univariate models to predict spacecraft prices.
    """

    try:
        csv_path = os.path.join(os.path.dirname(__file__), "spacecraft_data.csv")
        data = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("Error: spacecraft_data.csv not found!")
        print("Make sure the file is in the same directory as this script.")
        return 1
    
    Y = np.array(data[['Sell_price']])
    
    # ======================== 1 ================================

    X_age = np.array(data[['Age']])
    myLR_age, mse_age = train_and_evaluate_model(
        X=X_age,
        Y=Y,
        thetas_init=[[1000.0], [-1.0]],
        alpha=2.5e-5,
        max_iter=100000,
        feature_name='Age (in years)',
        model_name='Model 1: Sell Price vs Age'
    )
    
    X_thrust = np.array(data[['Thrust_power']])
    myLR_thrust, mse_thrust = train_and_evaluate_model(
        X=X_thrust,
        Y=Y,
        thetas_init=[[0.0], [1.0]],
        alpha=1e-4,
        max_iter=100000,
        feature_name='Thrust power (in 10 Km/s)',
        model_name='Model 2: Sell Price vs Thrust Power'
    )
    
    X_distance = np.array(data[['Terameters']])
    myLR_distance, mse_distance = train_and_evaluate_model(
        X=X_distance,
        Y=Y,
        thetas_init=[[700.0], [-1.0]],
        alpha=1e-4,
        max_iter=100000,
        feature_name='Distance total (in Tmeters)',
        model_name='Model 3: Sell Price vs Total Distance'
    )
    
    print(f"\n{'='*70}")
    print("                      PART 1 : Univariate Linear Regression")
    print(f"{'='*70}")
    print(f"{'Model':<30} {'MSE':<20} {'θ0':<15} {'θ1'}")
    print(f"{'-'*70}")
    print(f"{'1. Age':<30} {mse_age:<20.2f} {myLR_age.thetas[0][0]:<15.4f} {myLR_age.thetas[1][0]:.4f}")
    print(f"{'2. Thrust Power':<30} {mse_thrust:<20.2f} {myLR_thrust.thetas[0][0]:<15.4f} {myLR_thrust.thetas[1][0]:.4f}")
    print(f"{'3. Total Distance':<30} {mse_distance:<20.2f} {myLR_distance.thetas[0][0]:<15.4f} {myLR_distance.thetas[1][0]:.4f}")
    print(f"{'-'*70}")
    
    models = [
        ('Age', mse_age),
        ('Thrust Power', mse_thrust),
        ('Total Distance', mse_distance)
    ]
    best_model = min(models, key=lambda x: x[1])
    print(f"\n✓ Best univariate model: {best_model[0]} (lowest MSE: {best_model[1]:.2f})")
    
    # ======================== 2 ================================

    X = np.array(data[['Age', 'Thrust_power', 'Terameters']])
    feature_names = ['Age', 'Thrust_power', 'Terameters']
    feature_des = ['x1: age (in years)', 'x2: thrust power (in 10Km/s)', 'x3: distance totalizer value of spacecraft (in Tmeters)']
    model, mse = train_multivariate_model(X, Y, feature_names, feature_des, "Multivariate Linear Regression")
    
    print("\n" + "=" * 80)
    print(" " * 27 + "PART 2 : Multivariate Linear Regression")
    print("=" * 80)
    header = f"{'Model':<20} {'MSE':<15} {'θ0':<12} {'θ_age':<12} {'θ_thrust':<12} {'θ_terameters':<12}"
    print(header)
    print("-" * 80)
    row = f"{'4. Multivariate':<20} {mse:<15.4f} {model.thetas[0][0]:<12.4f} {model.thetas[1][0]:<12.4f} {model.thetas[2][0]:<12.4f} {model.thetas[3][0]:<12.4f}"
    print(row)
    print("-" * 80)

    models = [
        ('Age', mse_age),
        ('Thrust Power', mse_thrust),
        ('Total Distance', mse_distance),
        ('Multi', mse)
    ]
    best_model = min(models, key=lambda x: x[1])
    print(f"\n✓ Best model: {best_model[0]} (lowest MSE: {best_model[1]:.2f})")

    return 0

if __name__ == "__main__":
    main()