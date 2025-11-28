import numpy as np

def data_spliter(x, y, proportion):
    """Shuffles and splits the dataset (given by x and y) into a training and a test set,
    while respecting the given proportion of examples to be kept in the training set.
    Args:
        x: has to be an numpy.array, a matrix of dimension m * n.
        y: has to be an numpy.array, a vector of dimension m * 1.
        proportion: has to be a float, the proportion of the dataset that will be assigned to the
        training set.
    Return:
        (x_train, x_test, y_train, y_test) as a tuple of numpy.array
        None if x or y is an empty numpy.array.
        None if x and y do not share compatible dimensions.
        None if x, y or proportion is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        return None
    if x.size == 0 or y.size == 0:
        return None
    if x.shape[0] != y.shape[0]:
        return None
    if not isinstance(proportion, float) or not (0 < proportion < 1):
        return None

    # Mélanger les indices
    m = x.shape[0]
    indices = np.arange(m)
    np.random.shuffle(indices)

    # Appliquer le shuffle aux datasets
    x_shuffled = x[indices]
    y_shuffled = y[indices]

    # Découper selon la proportion
    train_size = int(m * proportion)
    x_train = x_shuffled[:train_size]
    x_test = x_shuffled[train_size:]
    y_train = y_shuffled[:train_size]
    y_test = y_shuffled[train_size:]

    return x_train, x_test, y_train, y_test

def main():
    """Tester of my splitter of data function"""

    print("============= TEST ===================")

    x1 = np.array([1, 42, 300, 10, 59]).reshape((-1, 1))
    y = np.array([0, 1, 0, 1, 0]).reshape((-1, 1))
    print("x1 array : \n", x1)
    print("y array : \n", y)

    print("============= 1 ===================")
    x1_train, x1_test, y_train, y_test = data_spliter(x1, y, 0.8)
    print("x1 train:", x1_train, "\n", "x1 test:", x1_test, "\n", "y train:", y_train, "\n", "y test:", y_test)
    print("Expected : (array([ 1, 59, 42, 300]), array([10]), array([0, 0, 1, 0]), array([1]))")

    print("============= 2 ===================")
    x1_train, x1_test, y_train, y_test = data_spliter(x1, y, 0.5)
    print("x1 train:", x1_train, "\n", "x1 test:", x1_test, "\n", "y train:", y_train, "\n", "y test:", y_test)
    print("Expected : (array([59, 10]), array([ 1, 300, 42]), array([0, 1]), array([0, 0, 1]))")

    x2 = np.array([[ 1, 42],
                    [300, 10],
                    [ 59, 1],
                    [300, 59],
                    [ 10, 42]])
    print("x2 array : \n", x2)

    print("============= 3 ===================")
    x2_train, x2_test, y_train, y_test = data_spliter(x2, y, 0.8)
    print("x2 train:", x2_train, "\n", "x2 test:", x2_test, "\n", "y train:", y_train, "\n", "y test:", y_test)
    print("Expected : (array([[ 10, 42],\n\t\t[300, 59],\n\t\t[ 59, 1],\n\t\t[300, 10]]),\n\t\tarray([[ 1, 42]]),\n\t\tarray([0, 1, 0, 1]),\n\t\tarray([0]))")

    print("============= 4 ===================")
    x2_train, x2_test, y_train, y_test = data_spliter(x2, y, 0.5)
    print("x2 train:", x2_train, "\n", "x2 test:", x2_test, "\n", "y train:", y_train, "\n", "y test:", y_test)
    print("Expected : (array([[59, 1],\n\t\t[10, 42]]),\n\t\tarray([[300, 10],\n\t\t[300, 59],\n\t\t[ 1, 42]]),\n\t\tarray([0, 0]),\n\t\tarray([1, 1, 0]))")

    return 0

if __name__ == "__main__":
    main()

# BUT :
# splitter un dataset en deux parties (train et test), mais de manière aléatoire,
# tout en conservant les correspondances entre x et y
# POURQUOI :
# de telle sorte a reduire le biais de selection des donnees d'entrainement et de test
# ce biais et appele l'overfitting (sur-apprentissage) ou le bruit
# on veut eviter que le modele apprenne par coeur les donnees d'entrainement
# et qu'il soit incapable de generaliser sur des donnees qu'il n'a jamais vues