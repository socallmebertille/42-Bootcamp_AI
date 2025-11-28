import pandas as pd
import numpy as np
# from sklearn.metrics import mean_squared_error

from my_linear_regression import MyLinearRegression as MyLR

def main():
    """Tester of my functions of precision indicator"""
    print("============= TEST ===================")
    data = pd.read_csv("/home/sarah/bootcamp_ML/Module06/ex04/are_blue_pills_magics.csv")
    Xpill = np.array(data['Micrograms']).reshape(-1,1)
    Yscore = np.array(data['Score']).reshape(-1,1)
    
    # Modèles avec thetas fixes pour comparaison
    linear_model1 = MyLR(np.array([[89.0], [-8]]))
    linear_model2 = MyLR(np.array([[89.0], [-6]]))
    
    # Modèle qu'on va entraîner
    linear_model_trained = MyLR(np.array([[89.0], [-8]]), alpha=0.005, max_iter=100000)
    linear_model_trained.fit_(Xpill, Yscore)
    
    print("Thetas du modèle entraîné :")
    print(linear_model_trained.thetas)
    
    Y_model1 = linear_model1.predict_(Xpill)
    Y_model2 = linear_model2.predict_(Xpill)
    Y_model_trained = linear_model_trained.predict_(Xpill)
    
    # Graphiques
    xlabel = "Quantity of blue pill (in micrograms)"
    ylabel = "Space driving score"
    
    linear_model1.plot_data(Xpill, Yscore, False, 
                            "Model 1: θ0=89, θ1=-8", xlabel, ylabel)
    linear_model2.plot_data(Xpill, Yscore, False, 
                            "Model 2: θ0=89, θ1=-6", xlabel, ylabel)
    linear_model_trained.plot_data(Xpill, Yscore, False, 
                                   "Trained Model", xlabel, ylabel)
    
    linear_model_trained.plot_loss(Xpill, Yscore, 10.0)
    
    # MSE
    print("\nMSE Model 1:", MyLR.mse(Yscore, Y_model1))
    print("Expected : 57.60304285714282")
    # print("sklearn:", mean_squared_error(Yscore, Y_model1))
    
    print("\nMSE Model 2:", MyLR.mse(Yscore, Y_model2))
    print("Expected : 232.16344285714285")
    # print("sklearn:", mean_squared_error(Yscore, Y_model2))
    
    print("\nMSE Trained Model:", MyLR.mse(Yscore, Y_model_trained))
    # print("sklearn:", mean_squared_error(Yscore, Y_model_trained))
    
    return 0

if __name__ == "__main__":
    main()

# Patient: number of the patient.
# Micrograms: quantity of blue pills patient has taken (in micrograms).
# Score: Standardized score at the spacecraft driving test.