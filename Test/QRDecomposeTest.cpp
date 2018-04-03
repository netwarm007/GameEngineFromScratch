#include <iomanip>
#include <iostream>
#include "MatrixComposeDecompose.hpp"

using namespace My;
using namespace std;

int main(int , char** )
{
    Matrix3X3f A = {{{
        {12.0f, 6.0f, -4.0f},
        {-51.0f, 167.0f, 24.0f},
        {4.0f, -68.0f, -41.0f},
    }}};
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