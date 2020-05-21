#pragma once
#include "geommath.hpp"

namespace My {
// infact our matrix is column-major, so it is RQ decompose ...
template <typename T, int ROWS, int COLS>
inline void MatrixQRDecompose(const Matrix<T, ROWS, COLS>& in_matrix,
                              Matrix<T, COLS, COLS>& Q,
                              Matrix<T, ROWS, COLS>& R) {
    static_assert(ROWS >= COLS,
                  "We can only QR Decompose a Matrix which ROWS >= COLS. (note "
                  "our matrix is column major)");
    Matrix<T, ROWS, COLS> U;
    std::memset(U, 0x00, sizeof(U));
    std::memset(R, 0x00, sizeof(R));
    for (int i = 0; i < ROWS; i++) {
        U[i] = in_matrix[i];
        for (int j = 0; j < i; j++) {
            T numerator, denominator;
            DotProduct(numerator, U[i], U[j]);
            denominator = Length(U[j]);
            auto coefficient = (denominator) ? numerator / denominator : 0;
            U[i] = U[i] - coefficient * U[j];
            R[i][j] = coefficient;
        }

        R[i][i] = Length(U[i]);
        if (R[i][i]) U[i] = U[i] / R[i][i];

        if (i < COLS) {
            Q[i] = U[i];
        }
    }
}

template <typename T>
inline void MatrixPolarDecompose(const Matrix<T, 3, 3>& in_matrix,
                                 Matrix<T, 3, 3>& U, Matrix<T, 3, 3>& P) {
    U = in_matrix;
    Matrix<T, 3, 3> U_inv;
    Matrix<T, 3, 3> U_pre;

    do {
        U_pre = U;
        U_inv = U;
        if (!InverseMatrix3X3f(U_inv)) assert(0);
        Matrix<T, 3, 3> U_inv_trans;
        Transpose(U_inv_trans, U_inv);
        U = (U + U_inv_trans) * (T)0.5;
    } while (U != U_pre);

    U_inv = U;
    if (!InverseMatrix3X3f(U_inv)) assert(0);
    P = in_matrix * U_inv;
}

inline void Matrix4X4fCompose(Matrix4X4f& matrix, const Vector3f& rotation,
                              const Vector3f& scalar,
                              const Vector3f& translation) {
    Matrix4X4f matrix_rotate_x, matrix_rotate_y, matrix_rotate_z, matrix_rotate;
    MatrixRotationX(matrix_rotate_x, rotation[0]);
    MatrixRotationY(matrix_rotate_y, rotation[1]);
    MatrixRotationZ(matrix_rotate_z, rotation[2]);
    matrix_rotate = matrix_rotate_x * matrix_rotate_y * matrix_rotate_z;
    Matrix4X4f matrix_scale;
    MatrixScale(matrix_scale, scalar);
    Matrix4X4f matrix_translation;
    MatrixTranslation(matrix_translation, translation);
    matrix = matrix_scale * matrix_rotate * matrix_translation;
}

inline void Matrix4X4fDecompose(const Matrix4X4f& matrix, Vector3f& rotation,
                                Vector3f& scalar, Vector3f& translation) {
    translation.Set({matrix[3][0], matrix[3][1], matrix[3][2]});

    // QR decompose the top-left 3x3 matrix
    Matrix3X3f bases = {{{matrix[0][0], matrix[0][1], matrix[0][2]},
                         {matrix[1][0], matrix[1][1], matrix[1][2]},
                         {matrix[2][0], matrix[2][1], matrix[2][2]}}};

    Matrix3X3f U, P;
    MatrixPolarDecompose(bases, U, P);

    float scale_x = P[0][0];
    float scale_y = P[1][1];
    float scale_z = P[2][2];

    // decompose the scale
    scalar.Set({scale_x, scale_y, scale_z});

    // decompose the rotation matrix
    float theta_x = std::atan2(U[1][2], U[2][2]);
    float theta_y = -std::asin(U[0][2]);
    float theta_z = std::atan2(U[0][1], U[0][0]);

    rotation.Set({theta_x, theta_y, theta_z});
}

template <typename T, int N>
T Determin(const Matrix<T, N, N>& matrix) {
    T result = 1;
    Matrix<T, N, N> Q, R;
    MatrixQRDecompose(matrix, Q, R);
    for (int i = 0; i < N; i++) {
        result *= R[i][i];
    }

    return result;
}

template <typename T>
void Matrix3X3EigenValues(Vector3f& eigen_values,
                          Matrix<T, 3, 3>& eigen_vectors,
                          const Matrix<T, 3, 3>& real_symmetric_matrix) {
    auto& A = real_symmetric_matrix;
    T p1 = pow(A[0][1], 2) + pow(A[0][2], 2) + pow(A[1][2], 2);
    if (p1 == 0) {
        // A is diagonal.
        eigen_values.Set({A[0][0], A[1][1], A[2][2]});
    } else {
        T q = Trace(A) / 3;
        T p2 = pow(A[0][0] - q, 2) + pow(A[1][1] - q, 2) + pow(A[2][2] - q, 2) +
               2 * p1;
        T p = sqrt(p2 / 6);
        Matrix<T, 3, 3> I;
        BuildIdentityMatrix(I);
        auto B = (1 / p) * (A - q * I);  // I is the identity matrix
        T r = Determin(B) / 2;

        // In exact arithmetic for a symmetric matrix  -1 <= r <= 1
        // but computation error can leave it slightly outside this range.
        T phi;
        if (r <= -1)
            phi = PI / 3;
        else if (r >= 1)
            phi = 0;
        else
            phi = acos(r) / 3;

        // the eigenvalues satisfy eig3 <= eig2 <= eig1
        eigen_values[0] = q + 2 * p * cos(phi);
        eigen_values[2] = q + 2 * p * cos(phi + (2 * PI / 3));
        eigen_values[1] =
            3 * q - eigen_values[0] -
            eigen_values[2];  // since trace(A) = eig1 + eig2 + eig3

        // now we calculate the order
        auto M1 = A - eigen_values[0] * I;
        auto M2 = A - eigen_values[1] * I;
        auto M3 = A - eigen_values[2] * I;
        eigen_vectors[0] = (M2 * M3)[0];
        eigen_vectors[1] = (M1 * M3)[1];
        eigen_vectors[2] = (M1 * M2)[2];
        Normalize(eigen_vectors[0]);
        Normalize(eigen_vectors[1]);
        Normalize(eigen_vectors[2]);
    }
}
}  // namespace My
