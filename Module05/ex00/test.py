from matrix import Matrix, Vector

def main():
    """Tester of my Matrix and Vector classes"""
    print("============= TEST 1  ===================")
    m1 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    print(m1)
    print("shape : ", m1.shape)
    print("expected : (3, 2)")

    print("transpose : ")
    print(m1.T())
    print("expected : Matrix([[0., 2., 4.], [1., 3., 5.]])")

    print("shape of transpose : ", m1.T().shape)
    print("expected : (2, 3)")

    print("============= TEST 2 ===================")
    m1 = Matrix([[0., 2., 4.], [1., 3., 5.]])
    print(m1)
    print("shape :", m1.shape)
    print("expected : (2, 3)")

    print("transpose :")
    print(m1.T())
    print("expected : Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])")

    print("shape of transpose : ", m1.T().shape)
    print("expected : (3, 2)")

    print("============= TEST 3 ===================")
    m1 = Matrix([[0.0, 1.0, 2.0, 3.0],
                [0.0, 2.0, 4.0, 6.0]])
    m2 = Matrix([[0.0, 1.0],
                [2.0, 3.0],
                [4.0, 5.0],
                [6.0, 7.0]])
    print("multipication de 2 matrices")
    print(m1, "\n*\n", m2, "\n=\n")
    print(m1 * m2)
    print("expected: Matrix([[28., 34.], [56., 68.]])")

    print("============= TEST 4 ===================")
    m1 = Matrix([[0.0, 1.0, 2.0],
                [0.0, 2.0, 4.0]])
    v1 = Vector([[1], [2], [3]])
    # v3 = Vector([[1, 2], [3, 4]])
    # print(v3)
    print("multipication d'1 matrice et 1 vecteur")
    print(m1, "\n*\n", v1, "\n=\n")
    print(m1 * v1)
    print("expected : Matrix([[8], [16]]) OR Vector([[8], [16]])")

    print("============= TEST 5 ===================")
    v1 = Vector([[1], [2], [3]])
    v2 = Vector([[2], [4], [8]])
    print("multipication de 2 vecteurs")
    print(v1, "\n+\n", v2, "\n=\n")
    print(v1 + v2)
    print("expected : Vector([[3],[6],[11]])")

    return 0

if __name__ == "__main__":
    main()

# Exemple de test sur terminal Python

# (42AI-Bootcamp) saberton in ~/42AI-Bootcamp on main λ python
# Python 3.13.9 (main, Nov 19 2025, 22:47:49) [Clang 21.1.4 ] on linux
# Type "help", "copyright", "credits" or "license" for more information.
# >>> from Module05.ex00.matrix import Matrix
# >>> m = Matrix([[1,2],[3,4]])
# >>> m
# Matrix([[1, 2], [3, 4]], shape=(2, 2))
# >>> print(m)
# 1 2
# 3 4
# >>> 