import numpy as np
import pandas as pd
import os

def main():
    """Tester of my polynomial function"""

    try:
        csv_path = os.path.join(os.path.dirname(__file__), "solar_system_census.csv")
        data = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("Error: solar_system_census.csv not found!")
        print("Make sure the file is in the same directory as this script.")
        return 1


    print("============= TEST ===================")

    print(" : \n", )
    print("Expected : ")

    print(" : \n", )
    print("Expected : ")

    return 0

if __name__ == "__main__":
    main()
