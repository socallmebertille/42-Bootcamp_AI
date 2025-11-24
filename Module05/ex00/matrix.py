class Matrix:
    def __init__(self, data):
        if isinstance(data, tuple): # shape -> the matrix will be filled with zeros by default
            rows, cols = data
            self.data = [[0.0 for _ in range(cols)] for _ in range(rows)]
            self.shape = data
        elif isinstance(data, list): # list of list -> the elements of the matrix as a list
            if not all(isinstance(row, (list, int, float)) for row in data):
                raise ValueError("Data must be a list of lists")
            if all(isinstance(x, (int, float)) for x in data): # list of numbers (1D) -> convert to column vector
                self.data = [[x] for x in data]
                self.shape = (len(data), 1)
            elif all(isinstance(row, list) for row in data): # list of lists
                if not all(all(isinstance(x, (int, float)) for x in row) for row in data):
                    raise ValueError("All elements must be numbers")
                self.data = data
                self.shape = (len(data), len(data[0]) if len(data) > 0 else 0)
            else:
                raise ValueError("Invalid data format")
        else:
            raise ValueError("Invalid data type for Matrix initialization")
    
    def __add__(self, other):
        if not isinstance(other, Matrix):
            raise ValueError("Object is not a matrix")
        if self.shape != other.shape: # add : only matrices of same dimensions.
            raise ValueError("Matrices must have same dimensions")
        rows, cols = self.shape
        result = [
            [self.data[i][j] + other.data[i][j] for j in range(cols)]
            for i in range(rows)
        ]
        return Matrix(result)
    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if not isinstance(other, Matrix):
            raise ValueError("Object is not a matrix")
        if self.shape != other.shape: # sub : only matrices of same dimensions.
            raise ValueError("Matrices must have same dimensions")
        rows, cols = self.shape
        result = [
            [self.data[i][j] - other.data[i][j] for j in range(cols)]
            for i in range(rows)
        ]
        return Matrix(result)
    def __rsub__(self, other):
        return self.__sub__(other)

    def __truediv__(self, other):
        if not isinstance(other, (int, float)): # div : only scalars.
            raise ValueError("Can only divide by scalar")
        if other == 0: # div : by 0 -> forbidden function 
            raise ZeroDivisionError("Cannot divide by zero")
        rows, cols = self.shape
        result = [
            [self.data[i][j] / other for j in range(cols)]
            for i in range(rows)
        ]
        return Matrix(result)
    def __rtruediv__(self, other):
        raise ValueError("Cannot divide a scalar by a Matrix")

    def __mul__(self, other):
        rows, cols = self.shape
        if isinstance(other, (int, float)): # mult : by scalars.
            result = [
                [self.data[i][j] * other for j in range(cols)]
                for i in range(rows)
            ]
            return Matrix(result)
        if isinstance(other, Vector): # mult : by vectors.
            if self.shape[1] != other.shape[0]:
                raise ValueError("Matrix and Vector dimensions do not match")
            result = []
            for i in range(rows):
                s = 0.0
                for j in range(cols):
                    s += self.data[i][j] * other.data[j][0]
                result.append([s]) # convert to list of lists
            return Vector(result) # returns a Vector if we perform Matrix * Vector mutliplication.
        if isinstance(other, Matrix): # mult : by matrices.
            if self.shape[1] != other.shape[0]:
                raise ValueError("Matrix dimensions do not match")
            m, n = self.shape
            n2, p = other.shape
            result = [
                [
                    sum(self.data[i][k] * other.data[k][j] for k in range(n))
                    for j in range(p)
                ]
                for i in range(m)
            ]
            return Matrix(result)
        raise ValueError("Wrong type of multipicateur")
    def __rmul__(self, other):
        return self.__mul__(other)

    def __str__(self):
        rows, cols = self.shape
        if rows == 1:
            return " ".join(str(self.data[0][j]) for j in range(cols))
        if cols == 1:
            return "\n".join(str(self.data[i][0]) for i in range(rows))
        return "\n".join(" ".join(map(str, row)) for row in self.data)

    def __repr__(self):
        return f"Matrix({self.data}, shape={self.shape})"
    
    def T(self):
        rows, cols = self.shape
        result = [
            [self.data[i][j] for i in range(rows)]
            for j in range(cols)
        ]
        return Matrix(result)

class Vector(Matrix):
    def __init__(self, data):
        if all(isinstance(x, (int, float)) for x in data): # convert vector into list of lists
            data = [[x] for x in data]
        elif all(isinstance(row, list) and len(row) == 1 for row in data): # column vector
            pass
        elif len(data) == 1 and isinstance(data[0], list): # row vector
            pass
        else:
            raise ValueError("Vector must be a row or column")
        super().__init__(data)

    def dot(self, v: "Vector"):
        if self.shape != v.shape:
            raise ValueError("Vectors must have same shape")
        return sum(self.data[i][0] * v.data[i][0] for i in range(self.shape[0]))