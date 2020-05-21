#include <functional>
#include <iostream>
#include <random>

#include "Linear.hpp"
#include "geommath.hpp"

using namespace My;
using namespace std;

int main(int argc, char** argv) {
    int interpolate_count = 100;

    if (argc > 1) {
        interpolate_count = atoi(argv[1]);
    }

    default_random_engine generator;
    generator.seed(48);
    uniform_real_distribution<float> distribution_r(-1.0f * PI, 1.0f * PI);
    uniform_real_distribution<float> distribution_s(0.1f, 100.0f);
    uniform_real_distribution<float> distribution_t(-1000.0f, 1000.0f);
    auto dice_r = bind(distribution_r, generator);
    auto dice_s = bind(distribution_s, generator);
    auto dice_t = bind(distribution_t, generator);

    // generate start point matrix
    Vector3f translation_1({dice_t(), dice_t(), dice_t()});
    Vector3f scale_1({dice_s(), dice_s(), dice_s()});
    Vector3f rotation_1({dice_r(), dice_r(), dice_r()});
    Matrix4X4f matrix_transform_1;
    Matrix4X4fCompose(matrix_transform_1, rotation_1, scale_1, translation_1);

    Vector3f v({dice_t(), dice_t(), dice_t()});

    // generate end point matrix
    Vector3f translation_2({dice_t(), dice_t(), dice_t()});
    Vector3f scale_2({dice_s(), dice_s(), dice_s()});
    Vector3f rotation_2({dice_r(), dice_r(), dice_r()});
    Matrix4X4f matrix_transform_2;
    Matrix4X4fCompose(matrix_transform_2, rotation_2, scale_2, translation_2);

    auto v1 = v;
    TransformCoord(v1, matrix_transform_1);
    auto v2 = v;
    TransformCoord(v2, matrix_transform_2);

    cout << "Start Point:" << v1;
    cout << "End Point:" << v2;
    cout << "_________________" << endl;
    cout << "Start Translation: " << translation_1;
    cout << "End Translation: " << translation_2;
    cout << "_________________" << endl;
    cout << "Start Scalar: " << scale_1;
    cout << "End Scalar: " << scale_2;
    cout << "_________________" << endl;
    cout << "Start Rotation: " << rotation_1;
    cout << "End Rotation: " << rotation_2;
    cout << "=================" << endl;

    Linear<Matrix4X4f, float> linear_introplator(
        {matrix_transform_1, matrix_transform_2});

    cout << endl;
    cout << "Interpolate: " << endl;
    for (int i = 0; i <= interpolate_count; i++) {
        auto inter_matrix =
            linear_introplator.Interpolate(i * 1.0f / interpolate_count, 1);
        Vector3f rotation, scalar, translation;
        Matrix4X4fDecompose(inter_matrix, rotation, scalar, translation);
        auto v_inter = v;
        TransformCoord(v_inter, inter_matrix);
        cout << "#" << i << endl;
        cout << "_________________" << endl;
        cout << "Interpolated Position: " << v_inter;
        cout << "_________________" << endl;
        cout << "Rotation: " << rotation;
        cout << "Scalar: " << scalar;
        cout << "Translation: " << translation;
        cout << "=================" << endl;
    }

    return 0;
}