import numpy as np
import pandas as pd
import pickle
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ex06.ridge import MyRidge as MyR

def main():
    """Tester of my Machine Learning function"""

    try:
        csv_path = os.path.join(os.path.dirname(__file__), "space_avocado.csv")
        data = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("Error: space_avocado.csv not found!")
        print("Make sure the file is in the same directory as this script.")
        return 1

    X = data[["weight", "prod_distance", "time_delivery"]].to_numpy()
    Y = data[["target"]].to_numpy()
    theta = np.ones((X.shape[1] + 2, 1))                                    # init de ma regression (theta_nb 1 pour biais)
    lr = MyR(theta, alpha=1e-3, max_iter=100000)

    # Step 1: Split data into train (64%), validation (16%), test (20%)
    # First split: 80% train+val, 20% test
    X_trainval, X_test, Y_trainval, Y_test = MyR.data_spliter_(X, Y, 0.8)
    # Second split: 80% of trainval = train, 20% of trainval = validation
    X_train, X_val, Y_train, Y_val = MyR.data_spliter_(X_trainval, Y_trainval, 0.8)

    lambdas = np.arange(0, 1.2, 0.2)  # Lambda from 0 to 1 with step 0.2
    models = {}
    best_model = None
    best_mse = float('inf')
    best_params = {}

    # Step 2-6: Train and evaluate all models
    for degree in range(1, 5):
        print(f"\n=== Training degree {degree} ===")

        X_poly = MyR.add_polynomial_features(X_train, degree)                 # step 2 : polynomial features
        X_poly_val = MyR.add_polynomial_features(X_val, degree)
        X_poly_test = MyR.add_polynomial_features(X_test, degree)

        X_poly_train_norm, mean, std = MyR.normalize_features(X_poly)         # step 3 : normalisation
        X_poly_val_norm = (X_poly_val - mean) / std
        X_poly_test_norm = (X_poly_test - mean) / std

        for lambda_ in lambdas:
            print(f"\n  Training degree={degree}, lambda={lambda_:.1f}...", end=" ")
            theta = np.ones((X_poly_train_norm.shape[1] + 1, 1))
            lr = MyR(theta, alpha=1e-3, max_iter=100000, lambda_=lambda_)
            
            lr.fit_(X_poly_train_norm, Y_train)                              # step 4 : entrainement
            
            y_pred_train = lr.predict_(X_poly_train_norm)                    # step 5 : evaluation
            y_pred_val = lr.predict_(X_poly_val_norm)
            y_pred_test = lr.predict_(X_poly_test_norm)
            
            mse_train = lr.mse(Y_train, y_pred_train)                        # step 6 : test performance
            mse_val = lr.mse(Y_val, y_pred_val)
            mse_test = lr.mse(Y_test, y_pred_test)
            
            print(f"Train MSE: {mse_train:.2f}, Val MSE: {mse_val:.2f}")
            
            model_key = f"degree_{degree}_lambda_{lambda_:.1f}"              # step 7 : save model
            models[model_key] = {
                'degree': degree,
                'lambda': lambda_,
                'theta': lr.thetas.copy(),
                'mean': mean,
                'std': std,
                'mse_train': mse_train,
                'mse_val': mse_val,
                'mse_test': mse_test
            }
            
            if mse_val < best_mse:                      # Evaluate models on the cross-validation set
                best_mse = mse_val
                best_model = model_key
                best_params = {
                    'degree': degree,
                    'lambda': lambda_,
                    'theta': lr.thetas.copy(),
                    'mean': mean,
                    'std': std
                }

    # Save all models
    with open(os.path.join(os.path.dirname(__file__), "models.pkl"), "wb") as f:
        pickle.dump(models, f)
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"{'='*60}")
    print(f"\nBest model: {best_model}")
    print(f"  Degree: {best_params['degree']}")
    print(f"  Lambda: {best_params['lambda']:.1f}")
    print(f"  Validation MSE: {best_mse:.2f}")
    print(f"  Test MSE: {models[best_model]['mse_test']:.2f}")
    print(f"\nAll models saved to 'models.pkl'")
    
    # Save best model info separately
    with open(os.path.join(os.path.dirname(__file__), "best_model.pkl"), "wb") as f:
        pickle.dump({
            'model_key': best_model,
            'params': best_params,
            'mse_val': best_mse
        }, f)

    return 0

if __name__ == "__main__":
    main()