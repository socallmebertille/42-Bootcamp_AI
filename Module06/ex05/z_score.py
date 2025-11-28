import numpy as np

def zscore(x):
    """Computes the normalized version of a non-empty numpy.ndarray using the z-score standardization.
    Args:
        x: has to be an numpy.ndarray, a vector.
    Returns:
        x’ as a numpy.ndarray.
        None if x is a non-empty numpy.ndarray or not a numpy.ndarray.
    Raises:
        This function shouldn’t raise any Exception.
    """
    if not isinstance(x, np.ndarray):
        return None
    if x.size == 0:
        return None
    if x.ndim < 1:
        return None
    x_flat = x.reshape(-1)
    mu = np.mean(x_flat)
    sigma = np.std(x_flat)
    if sigma == 0:
        return None
    z = (x_flat - mu) / sigma
    return z.reshape(x.shape)

def main():
    """Tester of my functions of precision indicator"""

    print("============= TEST ===================")

    X = np.array([0, 15, -9, 7, 12, 3, -21])

    print("x array : ", X)

    print("zscore : ", zscore(X))
    print("Expected : array([-0.08620324, 1.2068453 , -0.86203236, 0.51721942, 0.94823559, 0.17240647, -1.89647119])")

    Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    print("y array : ", Y)

    print("zscore : ", zscore(Y) )
    print("Expected : array([ 0.11267619, 1.16432067, -1.20187941, 0.37558731, 0.98904659, 0.28795027, -1.72770165])")

    return 0

if __name__ == "__main__":
    main()

# La normalisation z-score transforme les données de sorte que :
#   - la moyenne devient 0
#   - l’écart-type devient 1
#   - MAIS résultats non bornés