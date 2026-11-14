#include <cassert>
#include <iostream>
#include <chrono>
#include <array>
#include <random>
#include <stdexcept>
#include <string_view>

#include "MatrixComposeDecompose.hpp"
#include "geommath.hpp"

using namespace std;
using namespace My;

void require_near(float actual, double expected, const char* operation) {
    if (!std::isfinite(actual) ||
        std::abs(actual - expected) > 1e-3 + 2e-5 * std::abs(expected)) {
        cerr << operation << ": expected " << expected << ", got " << actual << endl;
        throw std::runtime_error(operation);
    }
}

Matrix8X8f reference_dct(const Matrix8X8f& matrix, bool inverse) {
    Matrix8X8f result;
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            double sum = 0.0;
            for (int k = 0; k < 8; k++) {
                for (int l = 0; l < 8; l++) {
                    const int u = inverse ? k : i;
                    const int v = inverse ? l : j;
                    const int x = inverse ? i : k;
                    const int y = inverse ? j : l;
                    const double scale = 0.25 * (u == 0 ? std::sqrt(0.5) : 1.0) *
                        (v == 0 ? std::sqrt(0.5) : 1.0);
                    sum += scale * matrix[k][l] *
                        std::cos((2 * x + 1) * u * PI / 16.0) *
                        std::cos((2 * y + 1) * v * PI / 16.0);
                }
            }
            result[i][j] = static_cast<float>(sum);
        }
    }
    return result;
}

void regression_test() {
    std::mt19937 random(12345);
    std::uniform_real_distribution<float> values(-128.0f, 127.0f);
    for (int sample = 0; sample < 68; sample++) {
        Matrix8X8f input;
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                input[i][j] = sample == 0 ? 0.0f :
                    sample == 1 ? 23.0f :
                    sample == 2 ? (i == 3 && j == 5 ? 127.0f : 0.0f) :
                    sample == 3 ? ((i + j) % 2 ? -128.0f : 127.0f) : values(random);
            }
        }
        const auto actual = DCT8X8(input);
        const auto expected = reference_dct(input, false);
        const auto inverse = IDCT8X8(input);
        const auto expected_inverse = reference_dct(input, true);
        const auto reconstructed = IDCT8X8(actual);
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                require_near(actual[i][j], expected[i][j], "DCT reference");
                require_near(inverse[i][j], expected_inverse[i][j], "IDCT reference");
                require_near(reconstructed[i][j], input[i][j], "DCT round trip");
            }
        }
#ifdef USE_ISPC
        auto in_place = input;
        ispc::DCT8X8(in_place, in_place);
        ispc::IDCT8X8(in_place, in_place);
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                require_near(in_place[i][j], input[i][j], "in-place DCT round trip");
            }
        }
#endif
    }

    Vector<float, 37> a, b;
    double expected_dot = 0.0;
    for (int i = 0; i < 37; i++) {
        a[i] = static_cast<float>(i - 18);
        b[i] = static_cast<float>(i % 7 - 3);
        expected_dot += a[i] * b[i];
    }
    require_near(DotProduct(a, b), expected_dot, "multi-gang dot product");
#ifdef USE_ISPC
    for (size_t count : {size_t(0), size_t(1), size_t(3), size_t(4), size_t(8),
                         size_t(9), size_t(16), size_t(17), size_t(37)}) {
        float actual = -1.0f;
        double expected = 0.0;
        for (size_t i = 0; i < count; i++) expected += a[i] * b[i];
        ispc::DotProduct(a, b, &actual, count);
        require_near(actual, expected, "dot product tail");
    }
#endif
    Matrix4X4f transform = {{{1, 2, 3, 4}, {5, 6, 7, 8},
                             {9, 10, 11, 12}, {13, 14, 15, 16}}};
    Vector4f vector {1, 2, 3, 4};
    Transform(vector, transform);
    for (int i = 0; i < 4; i++) require_near(vector[i], 90 + 10 * i, "transform");
    Vector3f coordinate {1, 2, 3};
    MatrixTranslation(transform, 5.0f, 6.0f, 7.0f);
    TransformCoord(coordinate, transform);
    for (int i = 0; i < 3; i++) require_near(coordinate[i], 6 + 2 * i, "translation");
    cout << "GeomMath regression passed" << endl;
}

void dct_benchmark() {
    std::array<Matrix8X8f, 16> blocks;
    std::mt19937 random(12345);
    std::uniform_real_distribution<float> values(-128.0f, 127.0f);
    for (auto& block : blocks) {
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) block[i][j] = values(random);
        }
    }
    constexpr int iterations = 100000;
    for (bool inverse : {false, true}) {
        std::array<double, 5> timings;
        double checksum = 0.0;
        for (auto& timing : timings) {
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < iterations; i++) {
                auto& input = blocks[i % blocks.size()];
                input[0][0] += 0.001f;
                const auto output = inverse ? IDCT8X8(input) : DCT8X8(input);
                checksum += output[i % 8][(i / 8) % 8];
            }
            timing = std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - start).count() / iterations;
        }
        std::sort(timings.begin(), timings.end());
        cout << (inverse ? "IDCT" : "DCT") << " median us/block: " << timings[2]
             << ", checksum: " << checksum << endl;
    }
}

void vector_test() {
    Vector2f x {55.3f, 22.1f};
    cout << "Vector2f: ";
    cout << x;

    Vector3f a {1.0f, 2.0f, 3.0f};
    Vector3f b {5.0f, 6.0f, 7.0f};

    cout << "vec 1: ";
    cout << a;
    cout << "vec 2: ";
    cout << b;

    Vector3f c;
    CrossProduct(c, a, b);
    cout << "Cross Product of vec 1 and vec 2: ";
    cout << c;

    float d;
    DotProduct(d, a, b);
    cout << "Dot Product of vec 1 and vec 2: ";
    cout << d << endl;

    MulByElement(c, a, b);
    cout << "Element Product of vec 1 and vec 2: ";
    cout << c;

    DivByElement(a, c, b);
    cout << "Element Division of aboud product and vec 2: (should equal vec1) ";
    cout << a;

    Vector4f e {-3.0f, 3.0f, 6.0f, 1.0f};
    Vector4f f {2.0f, 0.0f, -0.7f, 0.0f};
    cout << "vec 3: " << e;
    cout << "vec 4: " << f;

    Vector4f g = e + f;
    cout << "vec 3 + vec 4: " << g;
    g = e - f;
    cout << "vec 3 - vec 4: " << g;

    Normalize(g);
    cout << "normalized: " << g;
}

void matrix_test() {
    Matrix4X4f m1;
    BuildIdentityMatrix(m1);

    cout << "Idendity Matrix: ";
    cout << m1;

    Matrix4X4f mEu;
    float yaw = 0.2f, pitch = 0.3f, roll = 0.4f;
    MatrixRotationYawPitchRoll(mEu, yaw, pitch, roll);

    cout << "Matrix of yaw(" << yaw << ") pitch(" << pitch << ") roll(" << roll
         << "):";
    cout << mEu;

    Matrix4X4f ry;
    float angle = PI / 2.0f;
    MatrixRotationY(ry, angle);

    cout << "Matrix of Rotation on Y(angle = " << angle << "):";
    cout << ry;

    Matrix4X4f rz;
    MatrixRotationZ(rz, angle);

    cout << "Matrix of Rotation on Z(angle = " << angle << "):";
    cout << rz;

    float x = 5.0f, y = 6.5f, z = -7.0f;
    Matrix4X4f translate;
    MatrixTranslation(translate, x, y, z);

    cout << "Matrix of Translation on X(" << x << ") Y(" << y << ") Z(" << z
         << "):";
    cout << translate;

    cout
        << "Matrix multiplication: Rotation Y * Rotation Z * Translation on X:";
    Matrix4X4f transform = m1 * ry * rz * translate;
    cout << transform;

    Vector3f rotation, scalar, translation;
    Matrix4X4fDecompose(transform, rotation, scalar, translation);
    cout << "Decompose of Transform Matrix: " << endl;
    cout << "Rotation: " << rotation;
    cout << "Scalar: " << scalar;
    cout << "Translation: " << translation;
    cout << endl;

    Matrix4X4f recomposed_transform;
    Matrix4X4fCompose(recomposed_transform, rotation, scalar, translation);
    cout << "Re-composed Transform Matrix: " << endl;
    cout << recomposed_transform;

    Vector3f v {1.0f, 0.0f, 0.0f};

    Vector3f v1 = v;
    cout << "Vector : " << v1;
    cout << "Transform by Rotation Y Matrix:";
    cout << ry;
    TransformCoord(v1, ry);
    cout << "Now the vector becomes: " << v1;
    cout << endl;

    v1 = v;
    cout << "Vector : " << v1;
    cout << "Transform by Rotation Z Matrix:";
    cout << rz;
    TransformCoord(v1, rz);
    cout << "Now the vector becomes: " << v1;
    cout << endl;

    v1 = v;
    cout << "Vector : " << v1;
    cout << "Transform by Translation Matrix:";
    cout << translate;
    TransformCoord(v1, translate);
    cout << "Now the vector becomes: " << v1;
    cout << endl;

    v1 = v;
    cout << "Vector : " << v1;
    cout << "Transform by Transform Matrix:";
    cout << transform;
    TransformCoord(v1, transform);
    cout << "Now the vector becomes: " << v1;
    cout << endl;

    Vector3f v2 = v;
    cout << "Vector : " << v2;
    cout << "Transform by Re-Composed Transform Matrix:";
    cout << recomposed_transform;
    TransformCoord(v2, recomposed_transform);
    cout << "Now the vector becomes: " << v2;
    cout << "Error between vector transformed by origin and recomposed "
            "transform:"
         << endl;
    cout << v1 - v2;
    cout << endl;
    assert(Length(v1 - v2) < 10E-6f);

    Vector3f position {0, 0, -5}, lookAt {0, 0, 0}, up {0, 1, 0};
    Matrix4X4f view;
    BuildViewRHMatrix(view, position, lookAt, up);
    cout << "View Matrix with position(" << position << ") lookAt(" << lookAt
         << ") up(" << up << "):";
    cout << view;

    float fov = PI / 2.0f, aspect = 16.0f / 9.0f, near = 1.0f, far = 100.0f;
    Matrix4X4f perspective;
    BuildPerspectiveFovLHMatrix(perspective, fov, aspect, near, far);
    cout << "(Left-Handed Coordinate System) Perspective Matrix with fov("
         << fov << ") aspect(" << aspect << ") near ... far(" << near << " ... "
         << far << "):";
    cout << perspective;

    BuildPerspectiveFovRHMatrix(perspective, fov, aspect, near, far);
    cout << "(Right-Handed Coordinate System) Perspective Matrix with fov("
         << fov << ") aspect(" << aspect << ") near ... far(" << near << " ... "
         << far << "):";
    cout << perspective;

    Matrix4X4f mvp = view * perspective;
    cout << "MVP: " << mvp;

    Matrix3X3f invertable3x3 = {{
        {1.0f, 1.0f, 0.0f},
        {0.0f, 2.0f, 0.0f},
        {0.0f, 0.0f, 1.0f},
    }};
    cout << "Known Invertable Matrix: " << invertable3x3;
    assert(InverseMatrix3X3f(invertable3x3));
    cout << "Inverse of Matrix: " << invertable3x3;

    Matrix4X4f invertable = {{{1.0f, 1.0f, 0.0f, 0.0f},
                              {0.0f, 2.0f, 0.0f, 0.0f},
                              {0.0f, 0.0f, 1.0f, 0.0f},
                              {13.0f, 14.0f, 15.0f, 1.0f}}};
    cout << "Known Invertable Matrix: " << invertable;
    assert(InverseMatrix4X4f(invertable));
    cout << "Inverse of Matrix: " << invertable;

    Matrix4X4f non_invertable = {{{1.0f, 2.0f, 3.0f, 4.0f},
                                  {5.0f, 6.0f, 7.0f, 8.0f},
                                  {9.0f, 10.0f, 11.0f, 12.0f},
                                  {13.0f, 14.0f, 15.0f, 16.0f}}};
    cout << "Known Sigular(Not Invertable) Matrix: " << non_invertable;
    assert(!InverseMatrix4X4f(non_invertable));
    cout << "InverseMatrix4X4f returns false." << endl;

    Matrix8X8f pixel_block = {{{-76, -73, -67, -62, -58, -67, -64, -55},
                               {-65, -69, -73, -38, -19, -43, -59, -56},
                               {-66, -69, -60, -15, 16, -24, -62, -55},
                               {-65, -70, -57, -6, 26, -22, -58, -59},
                               {-61, -67, -60, -24, -2, -40, -60, -58},
                               {-49, -63, -68, -58, -51, -60, -70, -53},
                               {-43, -57, -64, -69, -73, -67, -63, -45},
                               {-41, -49, -59, -60, -63, -52, -50, -34}}};
    cout << "A 8X8 int pixel block: " << pixel_block;
    Matrix8X8f pixel_block_dct = DCT8X8(pixel_block);
    cout << "After DCTII: " << pixel_block_dct;

    Matrix8X8f pixel_block_reconstructed = IDCT8X8(pixel_block_dct);
    cout << "After IDCTII: " << pixel_block_reconstructed;

    Matrix8X8f pixel_error = pixel_block_reconstructed - pixel_block;
    cout << "DCT-IDCT error: " << pixel_error;
}

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--benchmark") {
        dct_benchmark();
        return 0;
    }
    cout << fixed;

    regression_test();
    vector_test();
    matrix_test();

    return 0;
}
