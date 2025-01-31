Readme file for advanced linear algebra tasks

1. Determinant (determinant(matrix))
Calculates the determinant of a square matrix recursively.
Edge cases handled:
0×0 matrix → Returns 1.
1×1 matrix → Returns the single element.
2×2 matrix → Uses ad - bc.
Larger matrices → Uses Laplace expansion.
Error handling:
Not a list of lists → TypeError.
Not square → ValueError.

2. Minor Matrix (minor(matrix))
Computes the minor matrix, where each entry is the determinant of the submatrix obtained by removing a row and column.
Edge cases handled:
1×1 matrix → Returns [[1]].
Error handling:
Not a list of lists → TypeError.
Not square or empty → ValueError.

3. Cofactor Matrix (cofactor(matrix))
Computes the cofactor matrix by applying the sign pattern (-1)^(i+j) to each minor.
Uses minor(matrix) to get the minor matrix.
Error handling:
Not a list of lists → TypeError.
Not square or empty → ValueError.

4. Adjugate Matrix (adjugate(matrix))
Computes the adjugate (adjoint) matrix, which is the transpose of the cofactor matrix.
Uses cofactor(matrix) to first get the cofactor matrix, then transposes it.
Error handling:
Not a list of lists → TypeError.
Not square or empty → ValueError.

5. Inverse Matrix (inverse(matrix))
Computes the inverse of a square matrix using:
𝐴−1=1
det(𝐴)×adj(𝐴)
A −1 = det(A)1×adj(A)
Uses determinant(matrix) to check if the matrix is singular.
Uses adjugate(matrix), then divides by det(matrix).
Returns None if matrix is singular (i.e., det(A) = 0).
Error handling:
Not a list of lists → TypeError.
Not square or empty → ValueError.

6. Matrix Definiteness (definiteness(matrix))
Classifies a square symmetric matrix as:
Positive definite → All eigenvalues > 0.
Positive semi-definite → All eigenvalues ≥ 0.
Negative definite → All eigenvalues < 0.
Negative semi-definite → All eigenvalues ≤ 0.
Indefinite → Contains both positive and negative eigenvalues.
Uses np.linalg.eigvals(matrix) to find eigenvalues.
Ensures the matrix is symmetric using np.allclose(matrix, matrix.T).
Error handling:
Not a numpy.ndarray → TypeError.
Not square, empty, or not symmetric → None.
