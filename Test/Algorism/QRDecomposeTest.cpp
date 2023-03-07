#include <functional>
#include <iomanip>
#include <iostream>
#include <random>

#include "MatrixComposeDecompose.hpp"

using namespace My;
using namespace std;

int main(int, char**) {
    default_random_engine generator;
    generator.seed(48);
    uniform_real_distribution<float> distribution_r(-1.0f * PI, 1.0f * PI);
    uniform_real_distribution<float> distribution_s(0.1f, 100.0f);
    uniform_real_distribution<float> distribution_t(-1000.0f, 1000.0f);
    auto dice_r = bind(distribution_r, generator);
    auto dice_s = bind(distribution_s, generator);
    auto dice_t = bind(distribution_t, generator);

    Vector3f translation({dice_t(), dice_t(), dice_t()});
    Vector3f scale({dice_s(), dice_s(), dice_s()});
    Vector3f rotation({dice_r(), dice_r(), dice_r()});
    Matrix4X4f matrix;
    Matrix4X4fCompose(matrix, rotation, scale, translation);

    Matrix3X3f A = {{{matrix[0][0], matrix[0][1], matrix[0][2]},
                     {matrix[1][0], matrix[1][1], matrix[1][2]},
                     {matrix[2][0], matrix[2][1], matrix[2][2]}}};
    Matrix3X3f Q, R;

    MatrixQRDecompose(A, Q, R);

    cout.precision(4);
    cout.setf(ios::fixed);

    cout << "QR Decompose of matrix A: " << endl;
    cout << A;
    cout << "Q:" << endl;
    cout << Q;
    cout << "R:" << endl;
    cout << R;

    Matrix3X3f A_dash = R * Q;
    cout << "Q * R: " << A_dash;
    cout << "Error: " << A_dash - A;

    return 0;
}
