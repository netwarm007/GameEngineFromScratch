#pragma once
#include "geommath.hpp"

namespace My {
    // infact our matrix is column-major, so it is RQ decompose ...
    template <typename T, int ROWS, int COLS>
    inline void MatrixQRDecompose(const Matrix<T, ROWS, COLS>& in_matrix,
        Matrix<T, COLS, COLS>& Q,
        Matrix<T, ROWS, COLS>& R)
    {
        static_assert(ROWS >= COLS, "We can only QR Decompose a Matrix which ROWS >= COLS. (note our matrix is column major)");
        Matrix<T, ROWS, COLS> U;
        std::memset(U, 0x00, sizeof(U));
        std::memset(R, 0x00, sizeof(R));
        for (int i = 0; i < ROWS; i++)
        {
            U[i] = in_matrix[i];
            for (int j = 0; j < i; j++)
            {
                T numerator, denominator;
                DotProduct(numerator, U[i], U[j]);
                denominator = Length(U[j]);
                auto coefficient = (denominator)? numerator / denominator : 0;
                U[i] = U[i] - coefficient * U[j];
                R[i][j] = coefficient;
            }

            R[i][i] = Length(U[i]);
	    if (R[i][i])
            	U[i] = U[i] / R[i][i];

            if (i < COLS)
            {
                Q[i] = U[i];
            }
        }
    }

    template <typename T>
    inline void MatrixPolarDecompose(const Matrix<T, 3, 3>& in_matrix,
        Matrix<T, 3, 3>& U,
        Matrix<T, 3, 3>& P)
    {
	U = in_matrix;
	T detU = 1;
	Matrix<T, 3, 3> U_inv = U;
	
	do {
		// now we calculate the inverse of U
		U = (U + U_inv) * (T)0.5;
		U_inv = U;
		if (!InverseMatrix3X3f(U_inv)) return;

		auto D = U - U_inv;

		// we QR decompose D for acceleration
		Matrix<T, 3, 3> Q;
		Matrix<T, 3, 3> R;
		MatrixQRDecompose(D, Q, R);

		// now, since Q is a pure rotation matrix (special orthogonal matrix), its det is 1,
		// we can get the det(U) by det(R). And, R is a triangular matrix, so we can calculate
		// its det by its diagonal entries
		for (int i = 0; i < 3; i++)
		{
		    detU *= R[i][i];
		}

		std::cerr << detU << std::endl;
	} while(abs(detU) > T(10E-6));

	P = in_matrix * U_inv;
    }

    inline void Matrix4X4fCompose(Matrix4X4f& matrix, const Vector3f& rotation, const Vector3f& scalar, const Vector3f& translation)
    {
        Matrix4X4f matrix_rotate_x, matrix_rotate_y, matrix_rotate_z, matrix_rotate;
        MatrixRotationX(matrix_rotate_x, rotation[0]);
        MatrixRotationY(matrix_rotate_y, rotation[1]);
        MatrixRotationZ(matrix_rotate_z, rotation[2]);
        matrix_rotate = matrix_rotate_x * matrix_rotate_y * matrix_rotate_z;
        Matrix4X4f matrix_scale;
        MatrixScale(matrix_scale, scalar);
        Matrix4X4f matrix_translation;
        MatrixTranslation(matrix_translation, translation);
        matrix = matrix_rotate * matrix_scale * matrix_translation;
    }

    inline void Matrix4X4fDecompose(const Matrix4X4f& matrix, Vector3f& rotation, Vector3f& scalar, Vector3f& translation)
    {
        translation.Set({matrix[3][0], matrix[3][1], matrix[3][2]});

        // QR decompose the top-left 3x3 matrix
        Matrix3X3f bases = {{{
            {matrix[0][0], matrix[0][1], matrix[0][2]},
            {matrix[1][0], matrix[1][1], matrix[1][2]},
            {matrix[2][0], matrix[2][1], matrix[2][2]}
        }}};

        Matrix3X3f Q, R;
        MatrixQRDecompose(bases, Q, R);

        float scale_x = Length(R[0]);
        float scale_y = Length(R[1]);
        float scale_z = Length(R[2]);

        // decompose the scale
        scalar.Set({scale_x, scale_y, scale_z});

        // decompose the rotation matrix
        float theta_x = atan2(Q[1][2], Q[2][2]);
        float theta_y = -asinf(Q[0][2]);
        float theta_z = atan2(Q[0][1], Q[0][0]);
        
        rotation.Set({theta_x, theta_y, theta_z});
    }
}
