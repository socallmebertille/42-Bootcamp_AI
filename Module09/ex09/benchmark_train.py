import numpy as np
import pandas as pd
import pickle
import os,sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from Module09.ex08.my_logistic_regression import MyLogisticRegression as MyLR
from Module08.ex08.other_metrics import f1_score_

def one_vs_all_predict(classifiers, X):
    """Return predicted class using one-vs-all classifiers."""
    probs = [clf.predict_(X) for clf in classifiers]
    probs = np.hstack(probs)
    return np.argmax(probs, axis=1).reshape(-1, 1)

def main():

    # ------------------ Load data ------------------
    base = os.path.dirname(__file__)
    # Drop the CSV index column to keep only the 3 features and 1 target
    X = pd.read_csv(os.path.join(base, "solar_system_census.csv"), index_col=0).to_numpy()
    Y = pd.read_csv(
        os.path.join(base, "solar_system_census_planets.csv"), index_col=0
    ).to_numpy()

    # ------------------ Split data ------------------
    # 80% train+val / 20% test
    X_tv, X_test, Y_tv, Y_test = MyLR.data_spliter_(X, Y, 0.8)
    # 80% train / 20% val (performed on the previous train+val split)
    X_train, X_val, Y_train, Y_val = MyLR.data_spliter_(X_tv, Y_tv, 0.8)

    # ------------------ Polynomial features ------------------
    DEGREE = 3
    X_train_poly = MyLR.add_polynomial_features(X_train, DEGREE)
    X_val_poly = MyLR.add_polynomial_features(X_val, DEGREE)
    X_test_poly = MyLR.add_polynomial_features(X_test, DEGREE)

    X_train_norm, mean, std = MyLR.normalize_features(X_train_poly)
    X_val_norm = (X_val_poly - mean) / std
    X_test_norm = (X_test_poly - mean) / std

    # ------------------ Training ------------------
    lambdas = np.arange(0, 1.2, 0.2)
    models = {}

    for lambda_ in lambdas:
        print(f"\nTraining models with λ = {lambda_:.1f}")
        classifiers = []

        for cls in range(4):
            y_bin = (Y_train == cls).astype(int)

            theta = np.zeros((X_train_norm.shape[1] + 1, 1))
            lr = MyLR(
                theta,
                alpha=0.01,
                max_iter=3000,
                penality='l2',
                lambda_=lambda_
            )
            lr.fit_(X_train_norm, y_bin)
            classifiers.append(lr)

        # -------- Validation evaluation --------
        y_val_pred = one_vs_all_predict(classifiers, X_val_norm)
        f1_val = f1_score_(Y_val, y_val_pred)

        print(f"λ = {lambda_:.1f} → Validation F1 = {f1_val:.4f}")

        models[lambda_] = {
            'lambda': float(lambda_),
            'degree': DEGREE,
            'classifiers': classifiers,
            'f1_val': float(f1_val),
            'mean': mean,
            'std': std
        }

    # ------------------ Test scores ------------------
    print("\nTest F1 scores:")
    for lambda_, model in models.items():
        y_test_pred = one_vs_all_predict(model['classifiers'], X_test_norm)
        f1_test = f1_score_(Y_test, y_test_pred)
        model['f1_test'] = float(f1_test)
        print(f"λ = {lambda_:.1f} → Test F1 = {f1_test:.4f}")

    # ------------------ Save models ------------------
    with open(os.path.join(base, "models.pkl"), "wb") as f:
        pickle.dump(models, f)

    print("\nAll models saved to models.pkl")
    return 0


if __name__ == "__main__":
    main()
