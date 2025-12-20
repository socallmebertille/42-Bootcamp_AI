import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ex06.ridge import MyRidge as MyR

def plot_evaluation_curves(models):
    """Plot MSE vs models for different lambda values"""
    degrees = sorted(set(m['degree'] for m in models.values()))
    lambdas = sorted(set(m['lambda'] for m in models.values()))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: MSE vs Lambda for each degree
    for degree in degrees:
        train_mse = []
        val_mse = []
        lambda_vals = []
        
        for lambda_ in lambdas:
            key = f"degree_{degree}_lambda_{lambda_:.1f}"
            if key in models:
                train_mse.append(models[key]['mse_train'])
                val_mse.append(models[key]['mse_val'])
                lambda_vals.append(lambda_)
        
        ax1.plot(lambda_vals, val_mse, marker='o', label=f'Degree {degree}')
    
    ax1.set_xlabel('Lambda (Regularization Parameter)')
    ax1.set_ylabel('Validation MSE')
    ax1.set_title('Model Performance: Validation MSE vs Lambda')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: MSE comparison for all models
    model_names = []
    train_mses = []
    val_mses = []
    
    for key, model in sorted(models.items()):
        model_names.append(f"D{model['degree']}_λ{model['lambda']:.1f}")
        train_mses.append(model['mse_train'])
        val_mses.append(model['mse_val'])
    
    x = np.arange(len(model_names))
    width = 0.35
    
    ax2.bar(x - width/2, train_mses, width, label='Train MSE', alpha=0.8)
    ax2.bar(x + width/2, val_mses, width, label='Validation MSE', alpha=0.8)
    ax2.set_xlabel('Model (Degree_Lambda)')
    ax2.set_ylabel('MSE')
    ax2.set_title('Train vs Validation MSE for All Models')
    ax2.set_xticks(x[::4])  # Show every 4th label to avoid crowding
    ax2.set_xticklabels(model_names[::4], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('evaluation_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_predictions(X, Y, models, best_model_key):
    """Plot true prices vs predicted prices for different lambda values"""
    best_model = models[best_model_key]
    degree = best_model['degree']
    
    # Get all lambdas for the best degree
    lambdas = sorted(set(m['lambda'] for m in models.values() 
                         if m['degree'] == degree))
    
    # Create polynomial features for plotting
    X_poly = MyR.add_polynomial_features(X, degree)
    
    # Use the best model's normalization parameters
    mean = best_model['mean']
    std = best_model['std']
    X_poly_norm = (X_poly - mean) / std
    
    # For visualization, use a single feature (e.g., weight)
    # Sort by this feature for better line plotting
    sort_idx = np.argsort(X[:, 0].flatten())
    X_sorted = X[sort_idx]
    Y_sorted = Y[sort_idx]
    X_poly_sorted = X_poly[sort_idx]
    X_poly_norm_sorted = X_poly_norm[sort_idx]
    
    plt.figure(figsize=(12, 7))
    
    # Plot true prices
    plt.scatter(range(len(Y_sorted)), Y_sorted, 
                color='blue', alpha=0.5, label='True Price', s=30)
    
    # Plot predictions for each lambda
    colors = plt.cm.rainbow(np.linspace(0, 1, len(lambdas)))
    
    for lambda_, color in zip(lambdas, colors):
        key = f"degree_{degree}_lambda_{lambda_:.1f}"
        if key in models:
            model_data = models[key]
            
            # Create model with saved parameters
            theta = model_data['theta']
            lr = MyR(theta, lambda_=lambda_)
            
            # Predict
            y_pred = lr.predict_(X_poly_norm_sorted)
            
            # Plot
            label = f'λ={lambda_:.1f}'
            if key == best_model_key:
                label += ' (Best)'
                plt.plot(range(len(y_pred)), y_pred, 
                        color=color, linewidth=2.5, label=label, linestyle='--')
            else:
                plt.plot(range(len(y_pred)), y_pred, 
                        color=color, linewidth=1.5, label=label, alpha=0.7)
    
    plt.xlabel('Sample Index (sorted by weight)')
    plt.ylabel('Price (Trantorian Units)')
    plt.title(f'True vs Predicted Prices (Polynomial Degree {degree})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('predictions_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Load models and evaluate the best one"""
    
    # Load models
    try:
        models_path = os.path.join(os.path.dirname(__file__), "models.pkl")
        with open(models_path, "rb") as f:
            models = pickle.load(f)
        print(f"Loaded {len(models)} models from 'models.pkl'")
    except FileNotFoundError:
        print("Error: models.pkl not found!")
        print("Please run benchmark_train.py first.")
        return 1
    
    # Load best model info
    try:
        best_path = os.path.join(os.path.dirname(__file__), "best_model.pkl")
        with open(best_path, "rb") as f:
            best_info = pickle.load(f)
        best_model_key = best_info['model_key']
        print(f"\nBest model: {best_model_key}")
    except FileNotFoundError:
        # Find best model based on validation MSE
        best_model_key = min(models.keys(), key=lambda k: models[k]['mse_val'])
        print(f"\nBest model (by validation MSE): {best_model_key}")
    
    # Load data
    try:
        csv_path = os.path.join(os.path.dirname(__file__), "space_avocado.csv")
        data = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("Error: space_avocado.csv not found!")
        return 1
    
    X = data[["weight", "prod_distance", "time_delivery"]].to_numpy()
    Y = data[["target"]].to_numpy()
    
    # Get best model parameters
    best_model = models[best_model_key]
    degree = best_model['degree']
    lambda_ = best_model['lambda']
    
    print(f"\nBest Model Parameters:")
    print(f"  Degree: {degree}")
    print(f"  Lambda: {lambda_:.1f}")
    print(f"  Training MSE: {best_model['mse_train']:.2f}")
    print(f"  Validation MSE: {best_model['mse_val']:.2f}")
    print(f"  Test MSE: {best_model['mse_test']:.2f}")
        
    # Plot evaluation curves
    print("\nGenerating evaluation curves...")
    plot_evaluation_curves(models)
    
    # Plot predictions comparison
    print("Generating predictions comparison...")
    plot_predictions(X, Y, models, best_model_key)
    
    print("\nDone! Check the generated plots.")
    
    return 0

if __name__ == "__main__":
    main()
