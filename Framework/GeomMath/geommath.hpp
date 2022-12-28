#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <unordered_set>
#include <vector>

#include "config.h"
#include "portable.hpp"

#ifdef USE_ISPC
namespace ispc { /* namespace */
extern "C" {
void DotProduct(const float* a, const float* b, float* result,
                  const size_t count);
void CrossProduct(const float a[3], const float b[3], float result[3]);
void AddByElement(const float* a, const float* b, float* result,
                  const size_t count);
void SubByElement(const float* a, const float* b, float* result,
                  const size_t count);
void MulByElement(const float* a, const float* b, float* result,
                  const size_t count);
void MulByElementi16(const int16_t* a, const int16_t* b, int16_t* result,
                     const size_t count);
void MulByElementi32(const int32_t* a, const int32_t* b, int32_t* result,
                     const size_t count);
void DivByElement(const float* a, const float* b, float* result,
                  const size_t count);
void DivByElementi16(const int16_t* a, const int16_t* b, int16_t* result,
                     const size_t count);
void DivByElementi32(const int32_t* a, const int32_t* b, int32_t* result,
                     const size_t count);
void Normalize(const size_t count, float* v, float length);
void Transform(float vector[4], const float matrix[16]);
void Transpose(const float* a, float* r, const uint32_t row_count,
               const uint32_t column_count);
void BuildIdentityMatrix(float* data, const int32_t n);
void MatrixExchangeYandZ(float* data, const int32_t rows, const int32_t cols);
bool InverseMatrix3X3f(float matrix[9]);
bool InverseMatrix4X4f(float matrix[16]);
void DCT8X8(const float g[64], float G[64]);
void IDCT8X8(const float G[64], float g[64]);
void Absolute(float* result, const float* a, const size_t count);
void Pow(const float* v, const size_t count, const float exponent,
         float* result);
void Sqrt(const float* v, const size_t count, float* result);
} /* end extern C */
} /* namespace */
#endif

#ifndef PI
#define PI 3.14159265358979323846
#endif

#ifndef TWO_PI
#define TWO_PI 3.14159265358979323846 * 2.0
#endif

namespace My {
template <class T>
concept Dimension = std::integral<T>;

template <class T>
constexpr float normalize(T value) {
    return value < 0
               ? -static_cast<float>(value) / (std::numeric_limits<T>::min)()
               : static_cast<float>(value) / (std::numeric_limits<T>::max)();
}

template <class T, Dimension auto N>
struct Vector {
    T data[N];

    __host__ __device__ Vector() { std::memset(data, 0x00, sizeof(T) * N); }

    __host__ __device__ explicit Vector(const T val) {
        for (Dimension auto i = 0; i < N; i++) {
            data[i] = val;
        }
    }

    __host__ __device__ Vector(std::initializer_list<const T> list) : Vector() {
        size_t i = 0;
        assert(list.size() <= N);
        for (auto val : list) {
            data[i++] = val;
        }
    }

    __host__ __device__ operator T*() { return reinterpret_cast<T*>(this); };

    __host__ __device__ operator const T*() const { return reinterpret_cast<const T*>(this); }

    __host__ __device__ void Set(const T val) {
        for (Dimension auto i = 0; i < N; i++) {
            data[i] = val;
        }
    }

    __host__ __device__ void Set(const T* pval) { std::memcpy(data, pval, sizeof(T) * N); }

    __host__ __device__ void Set(std::initializer_list<const T> list) {
        size_t i = 0;
        for (auto val : list) {
            data[i++] = val;
        }
    }

    __host__ __device__ Vector operator-() const {
        Vector result;

        for (Dimension auto i = 0; i < N; i++) {
            result[i] = -data[i];
        }

        return result;
    }

    __host__ __device__ Vector& operator=(const T* pf) {
        Set(pf);
        return *this;
    }

    __host__ __device__ Vector& operator=(const T f) {
        Set(f);
        return *this;
    }

    __host__ __device__ Vector& operator=(const Vector& v) {
        std::memcpy(this, &v, sizeof(v));
        return *this;
    }

    __host__ __device__ Vector& operator+=(const Vector& v) {
        *this = *this + v;
        return *this;
    }

    __host__ __device__ Vector& operator-=(const Vector& v) {
        *this = *this - v;
        return *this;
    }

    __host__ __device__ Vector& operator/=(const T scalar) {
        *this = *this / scalar;
        return *this;
    }

    __host__ __device__ Vector& operator/=(const Vector& v) {
        *this = *this / v;
        return *this;
    }

    __host__ __device__ T& operator[](size_t index) { return data[index]; }

    __host__ __device__ [[nodiscard]] const T& operator[](size_t index) const {
        return data[index];
    }
};

template <class T>
using Vector2 = Vector<T, 2>;
using Vector2f = Vector2<float>;

template <class T>
using Vector3 = Vector<T, 3>;
using Vector3f = Vector3<float>;
using Vector3i16 = Vector3<int16_t>;
using Vector3i32 = Vector3<int32_t>;

template <class T>
using Vector4 = Vector<T, 4>;
using Vector4f = Vector4<float>;
using R8G8B8A8Unorm = Vector4<uint8_t>;
using Vector4i = Vector4<uint8_t>;

template <class T>
class Quaternion : public Vector<T, 4> {
   public:
    using Vector<T, 4>::Vector;
    Quaternion() = default;
    explicit Quaternion(const Vector<T, 4> rhs) {
        std::memcpy(this, &rhs, sizeof(Quaternion));
    }
};

template <class T, Dimension auto N>
std::ostream& operator<<(std::ostream& out, Vector<T, N> vector) {
    out.precision(4);
    out.setf(std::ios::fixed);
    out << "( ";
    for (uint32_t i = 0; i < N; i++) {
        out << vector.data[i] << ((i == N - 1) ? ' ' : ',');
    }
    out << ")" << std::endl;

    return out;
}

template <class T, Dimension auto N>
__host__ __device__ void VectorAdd(Vector<T, N>& result, const Vector<T, N>& vec1,
               const Vector<T, N>& vec2) {
#ifdef USE_ISPC
    ispc::AddByElement(vec1, vec2, result, N);
#else
    for (size_t i = 0; i < N; i++) {
        result[i] = vec1[i] + vec2[i];
    }
#endif
}

template <class T, Dimension auto N>
__host__ __device__ Vector<T, N> operator+(const Vector<T, N>& vec1, const Vector<T, N>& vec2) {
    Vector<T, N> result;
    VectorAdd(result, vec1, vec2);

    return result;
}

template <class T, Dimension auto N>
__host__ __device__ Vector<T, N> operator+(const Vector<T, N>& vec, const T scalar) {
    Vector<T, N> result(scalar);
    VectorAdd(result, vec, result);

    return result;
}

template <class T, Dimension auto N>
__host__ __device__ void VectorSub(Vector<T, N>& result, const Vector<T, N>& vec1,
               const Vector<T, N>& vec2) {
#ifdef USE_ISPC
    ispc::SubByElement(vec1, vec2, result, N);
#else
    for (size_t i = 0; i < N; i++) {
        result[i] = vec1[i] - vec2[i];
    }
#endif
}

template <class T, Dimension auto N>
__host__ __device__ Vector<T, N> operator-(const Vector<T, N>& vec1, const Vector<T, N>& vec2) {
    Vector<T, N> result;
    VectorSub(result, vec1, vec2);

    return result;
}

template <class T>
__host__ __device__ inline void CrossProduct(T& result, const Vector<T, 2>& vec1,
                         const Vector<T, 2>& vec2) {
    result = vec1[0] * vec2[1] - vec1[1] * vec2[0];
}

template <class T>
__host__ __device__ inline void CrossProduct(Vector<T, 3>& result, const Vector<T, 3>& vec1,
                         const Vector<T, 3>& vec2) {
#ifdef USE_ISPC
    ispc::CrossProduct(vec1, vec2, result);
#else
    result[0] = vec1[1] * vec2[2] - vec1[2] * vec2[1];
    result[1] = vec1[2] * vec2[0] - vec1[0] * vec2[2];
    result[2] = vec1[0] * vec2[1] - vec1[1] * vec2[0];
#endif
}

template <class T>
__host__ __device__ inline auto CrossProduct(const Vector<T, 3>& vec1,
                         const Vector<T, 3>& vec2) {
    Vector<T, 3> result; 

    CrossProduct(result, vec1, vec2);

    return result;
}

template <class T>
__host__ __device__ inline void DotProduct(T& result, const Vector3<T>& vec1, const Vector3<T>& vec2) {
    result = vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2];
}

template <class T, Dimension auto N>
__host__ __device__ inline void DotProduct(T& result, const Vector<T, N>& vec1,
                       const Vector<T, N>& vec2) {
#ifdef USE_ISPC
    ispc::DotProduct(vec1, vec2, &result,
                  N);
#else
    for (int i = 0; i < N; i++) {
        result += vec1[i] * vec2[i];
    }
#endif
}

template <class T, Dimension auto N>
__host__ __device__ inline T DotProduct(const Vector<T, N>& vec1, const Vector<T, N>& vec2) {
    T result;
    DotProduct(result, vec1, vec2);
    return result;
}

template <class T, Dimension auto N>
__host__ __device__ inline void MulByElement(Vector<T, N>& result, const Vector<T, N>& a,
                         const Vector<T, N>& b) {
#ifdef USE_ISPC
    ispc::MulByElement(a, b, result, N);
#else
    for (size_t i = 0; i < N; i++) {
        result[i] = a[i] * b[i];
    }
#endif
}

template <class T, Dimension auto N>
__host__ __device__ inline void MulByElement(Vector<T, N>& result, const Vector<T, N>& a,
                         const T scalar) {
    Vector<T, N> v(scalar);

    MulByElement(result, a, v);
}

template <class T, Dimension auto N>
__host__ __device__ Vector<T, N> operator*(const Vector<T, N>& vec, const T scalar) {
    Vector<T, N> result;
    MulByElement(result, vec, scalar);

    return result;
}

template <class T, Dimension auto N>
__host__ __device__ Vector<T, N> operator*(const T scalar, const Vector<T, N>& vec) {
    Vector<T, N> result;
    MulByElement(result, vec, scalar);

    return result;
}

template <class T, Dimension auto N>
__host__ __device__ Vector<T, N> operator*(const Vector<T, N>& vec1, const Vector<T, N>& vec2) {
    Vector<T, N> result;
    MulByElement(result, vec1, vec2);

    return result;
}

template <class T, Dimension auto N>
__host__ __device__ inline void DivByElement(Vector<T, N>& result, const Vector<T, N>& a,
                         const Vector<T, N>& b) {
#ifdef USE_ISPC
    ispc::DivByElement(a, b, result, N);
#else
    for (size_t i = 0; i < N; i++) {
        result[i] = a[i] / b[i];
    }
#endif
}

template <class T, Dimension auto N>
__host__ __device__ inline void DivByElement(Vector<T, N>& result, const Vector<T, N>& a,
                         const T scalar) {
    Vector<T, N> v(scalar);

    DivByElement(result, a, v);
}

template <class T, Dimension auto N> requires std::floating_point<T>
__host__ __device__ Vector<T, N> operator/(const Vector<T, N>& vec, const T scalar) {
    Vector<T, N> result;
    DivByElement(result, vec, scalar);

    return result;
}

template <class T, Dimension auto N>
__host__ __device__ Vector<T, N> operator/(const T scalar, const Vector<T, N>& vec) {
    Vector<T, N> result;
    DivByElement(result, vec, scalar);

    return result;
}

template <class T, Dimension auto N>
__host__ __device__ Vector<T, N> operator/(const Vector<T, N>& vec1, const Vector<T, N>& vec2) {
    Vector<T, N> result;
    DivByElement(result, vec1, vec2);

    return result;
}

template <class T>
inline constexpr T pow(const T base,
                                 const T exponent) {
    return std::pow(base, exponent);
}

template <class T, Dimension auto N>
Vector<T, N> pow(const Vector<T, N>& vec, const T exponent) {
    Vector<T, N> result;
#ifdef USE_ISPC
    ispc::Pow(vec, N, exponent, result);
#else
    for (size_t i = 0; i < N; i++) {
        result[i] = pow(vec[i], exponent);
    }
#endif
    return result;
}

template <class T>
inline constexpr T sqrt(const T base) {
    return std::sqrt(base);
}

template <class T, Dimension auto N>
Vector<T, N> sqrt(const Vector<T, N>& vec) {
    Vector<T, N> result;
#ifdef USE_ISPC
    ispc::Sqrt(vec, N, result);
#else
    for (size_t i = 0; i < N; i++) {
        result[i] = sqrt(vec[i]);
    }
#endif
    return result;
}

template <class T>
inline constexpr T fabs(const T data) {
    return std::fabs(data);
}

template <class T, Dimension auto N>
Vector<T, N> fabs(const Vector<T, N>& vec) {
    Vector<T, N> result;
#ifdef USE_ISPC
    ispc::Absolute(result, vec, N);
#else
    for (size_t i = 0; i < N; i++) {
        result[i] = fabs(vec[i]);
    }
#endif
    return result;
}

template <class T, Dimension auto N>
__host__ __device__ inline T LengthSquared(const Vector<T, N>& vec) {
    T result;
    DotProduct(result, vec, vec);
    return result;
}

template <class T, Dimension auto N>
__host__ __device__ inline constexpr T Length(const Vector<T, N>& vec) {
    auto length_squared = LengthSquared(vec);
    return static_cast<T>(std::sqrt(length_squared));
}

template <class T, Dimension auto N>
__host__ __device__ inline bool operator>=(const Vector<T, N>&& vec, const T scalar) {
    return Length(vec) >= scalar;
}

template <class T, Dimension auto N>
__host__ __device__ inline bool operator>(const Vector<T, N>&& vec, const T scalar) {
    return Length(vec) > scalar;
}

template <class T, Dimension auto N>
__host__ __device__ inline bool operator<=(const Vector<T, N>&& vec, const T scalar) {
    return Length(vec) <= scalar;
}

template <class T, Dimension auto N>
inline bool operator<(const Vector<T, N>&& vec, const T scalar) {
    return Length(vec) < scalar;
}

template <class T, Dimension auto N>
__host__ __device__ inline void Normalize(Vector<T, N>& a) {
    T length = Length(a);
#ifdef USE_ISPC
    ispc::Normalize(N, a, length);
#else
    if (!length) return;
    const T one_over_length = 1.0 / length;
    for (size_t index = 0; index < N; index++) {
        a[index] = static_cast<T>(a[index] * one_over_length);
    }
#endif
}

template <class T, Dimension auto N>
__host__ __device__ inline bool isNearZero(const Vector<T, N>& vec) {
    bool result = true;
    const auto s = 1e-8;

    for (Dimension auto i = 0; i < N; i++) {
        if (fabs(vec[i]) >= s) {
            result = false;
        }
    }

    return result;
}

template <class T, Dimension auto N>
__host__ __device__ inline Vector<T, N> Reflect(const Vector<T, N>& v, const Vector<T, N>& n) {
    return v - 2 * DotProduct(v, n) * n;
}

template <class T, Dimension auto N>
__host__ __device__ inline Vector<T, N> Refract(const Vector<T, N>& v, const Vector<T, N>& n,
                            T etai_over_etat) {
    T cos_theta = fmin(DotProduct(-v, n), 1.0);

    Vector<T, N> r_out_perp = etai_over_etat * (v + cos_theta * n);
    Vector<T, N> r_out_parallel = - (T)sqrt(fabs(1.0 - LengthSquared(r_out_perp))) * n;
    return r_out_perp + r_out_parallel;
}

///////////////////////////
// Matrix

template <class T, Dimension auto ROWS, Dimension auto COLS>
struct Matrix {
    Vector<T, COLS> data[ROWS];

    __host__ __device__ Vector<T, COLS>& operator[](Dimension auto row_index) {
        return data[row_index];
    }

    __host__ __device__ const Vector<T, COLS>& operator[](Dimension auto row_index) const {
        return data[row_index];
    }

    __host__ __device__ operator T*() { return (T*)&data; };
    __host__ __device__ operator const T*() const { return static_cast<const T*>(&data[0][0]); };

    __host__ __device__ Matrix& operator=(const T* _data) {
        std::memcpy(data, _data, sizeof(T) * COLS * ROWS);
        return *this;
    }

    __host__ __device__ Matrix& operator=(const Matrix& rhs) {
        std::memcpy(data, rhs, sizeof(Matrix));
        return *this;
    }

    [[nodiscard]] bool isOrthogonal() const {
        Matrix trans;
        Transpose(trans, *this);
        Matrix I;
        BuildIdentityMatrix(I);
        if (*this * trans == I) {
            return true;
        }

        return false;
    }
};

template <class T>
using Matrix3X3 = Matrix<T, 3, 3>;

template <class T>
using Matrix4X4 = Matrix<T, 4, 4>;

using Matrix3X3f = Matrix<float, 3, 3>;
using Matrix4X4f = Matrix<float, 4, 4>;
using Matrix8X8i = Matrix<int32_t, 8, 8>;
using Matrix8X8f = Matrix<float, 8, 8>;

template <class T, Dimension auto ROWS, Dimension auto COLS>
std::ostream& operator<<(std::ostream& out, Matrix<T, ROWS, COLS> matrix) {
    out << std::endl;
    for (Dimension auto i = 0; i < ROWS; i++) {
        out << matrix[i];
    }
    out << std::endl;

    return out;
}

template <class T, Dimension auto ROWS, Dimension auto COLS>
__host__ __device__ void MatrixAdd(Matrix<T, ROWS, COLS>& result,
               const Matrix<T, ROWS, COLS>& matrix1,
               const Matrix<T, ROWS, COLS>& matrix2) {
#ifdef USE_ISPC
    ispc::AddByElement(matrix1, matrix2, result, ROWS * COLS);
#else
    for (size_t i = 0; i < ROWS * COLS; i++) {
        result[i] = matrix1[i] + matrix2[i];
    }
#endif
}

template <class T, Dimension auto ROWS, Dimension auto COLS>
__host__ __device__ Matrix<T, ROWS, COLS> operator+(const Matrix<T, ROWS, COLS>& matrix1,
                                const Matrix<T, ROWS, COLS>& matrix2) {
    Matrix<T, ROWS, COLS> result;
    MatrixAdd(result, matrix1, matrix2);

    return result;
}

template <class T, Dimension auto ROWS, Dimension auto COLS>
__host__ __device__ void MatrixSub(Matrix<T, ROWS, COLS>& result,
               const Matrix<T, ROWS, COLS>& matrix1,
               const Matrix<T, ROWS, COLS>& matrix2) {
#ifdef USE_ISPC
    ispc::SubByElement(matrix1, matrix2, result, ROWS * COLS);
#else
    for (size_t i = 0; i < ROWS * COLS; i++) {
        result[i] = matrix1[i] - matrix2[i];
    }
#endif
}

template <class T, Dimension auto ROWS, Dimension auto COLS>
__host__ __device__ void MatrixMulByElement(Matrix<T, ROWS, COLS>& result,
                        const Matrix<T, ROWS, COLS>& matrix1,
                        const Matrix<T, ROWS, COLS>& matrix2) {
#ifdef USE_ISPC
    ispc::MulByElement(matrix1, matrix2, result, ROWS * COLS);
#else
    for (size_t i = 0; i < ROWS * COLS; i++) {
        result[i] = matrix1[i] * matrix2[i];
    }
#endif
}

template <Dimension auto ROWS, Dimension auto COLS>
__host__ __device__ void MatrixMulByElementi32(Matrix<int32_t, ROWS, COLS>& result,
                           const Matrix<int32_t, ROWS, COLS>& matrix1,
                           const Matrix<int32_t, ROWS, COLS>& matrix2) {
#ifdef USE_ISPC
    ispc::MulByElementi32(matrix1, matrix2, result, ROWS * COLS);
#else
    for (size_t i = 0; i < ROWS * COLS; i++) {
        result[i] = matrix1[i] * matrix2[i];
    }
#endif
}

template <class T, Dimension auto ROWS, Dimension auto COLS>
__host__ __device__ Matrix<T, ROWS, COLS> operator-(const Matrix<T, ROWS, COLS>& matrix1,
                                const Matrix<T, ROWS, COLS>& matrix2) {
    Matrix<T, ROWS, COLS> result;
    MatrixSub(result, matrix1, matrix2);

    return result;
}

template <class T, Dimension auto Da, Dimension auto Db, Dimension auto Dc>
__host__ __device__ void MatrixMultiply(Matrix<T, Da, Dc>& result, const Matrix<T, Da, Db>& matrix1,
                    const Matrix<T, Dc, Db>& matrix2) {
    Matrix<T, Dc, Db> matrix2_transpose;
    Transpose(matrix2_transpose, matrix2);
    for (Dimension auto i = 0; i < Da; i++) {
        for (Dimension auto j = 0; j < Dc; j++) {
            DotProduct(result[i][j], matrix1[i], matrix2_transpose[j]);
        }
    }
}

template <class T, Dimension auto ROWS, Dimension auto COLS>
__host__ __device__ Matrix<T, ROWS, COLS> operator*(const Matrix<T, ROWS, COLS>& matrix1,
                                const Matrix<T, ROWS, COLS>& matrix2) {
    Matrix<T, ROWS, COLS> result;
    MatrixMultiply(result, matrix1, matrix2);

    return result;
}

template <class T, Dimension auto ROWS, Dimension auto COLS>
__host__ __device__ Matrix<T, ROWS, COLS> operator*(const Matrix<T, ROWS, COLS>& matrix,
                                const T scalar) {
    Matrix<T, ROWS, COLS> result;

    for (Dimension auto i = 0; i < ROWS; i++) {
        result[i] = matrix[i] * scalar;
    }

    return result;
}

template <class T, Dimension auto ROWS, Dimension auto COLS>
__host__ __device__ Matrix<T, ROWS, COLS> operator*(const T scalar,
                                const Matrix<T, ROWS, COLS>& matrix) {
    return matrix * scalar;
}

template <class T, Dimension auto ROWS1, Dimension auto COLS1,
          Dimension auto ROWS2, Dimension auto COLS2>
void Shrink(Matrix<T, ROWS1, COLS1>& matrix1,
            const Matrix<T, ROWS2, COLS2>& matrix2) {
    static_assert(
        ROWS1 < ROWS2,
        "[Error] Target matrix ROWS must smaller than source matrix ROWS!");
    static_assert(
        COLS1 < COLS2,
        "[Error] Target matrix COLS must smaller than source matrix COLS!");

    const size_t size = sizeof(T) * COLS1;
    for (Dimension auto i = 0; i < ROWS1; i++) {
        std::memcpy(matrix1[i], matrix2[i], size);
    }
}

template <class T, Dimension auto ROWS, Dimension auto COLS>
__host__ __device__ void Absolute(Matrix<T, ROWS, COLS>& result,
              const Matrix<T, ROWS, COLS>& matrix) {
#ifdef USE_ISPC
    ispc::Absolute(result, matrix, ROWS * COLS);
#else
    for (size_t i = 0; i < ROWS * COLS; i++) {
        result[i] = fabs(matrix[i]);
    }
#endif
}

template <class T, Dimension auto ROWS, Dimension auto COLS>
__host__ __device__ inline bool AlmostZero(const Matrix<T, ROWS, COLS>& matrix) {
    bool result = true;
    for (Dimension auto i = 0; i < ROWS; i++) {
        for (Dimension auto j = 0; j < COLS; j++) {
            if (std::abs(matrix[i][j]) > std::numeric_limits<T>::epsilon()) {
                result = false;
            }
        }
    }

    return result;
}

template <class T, Dimension auto ROWS, Dimension auto COLS>
__host__ __device__ inline bool operator==(const Matrix<T, ROWS, COLS>& matrix1,
                       const Matrix<T, ROWS, COLS>& matrix2) {
    return AlmostZero(matrix1 - matrix2);
}

template <class T, Dimension auto ROWS, Dimension auto COLS>
__host__ __device__ inline bool operator!=(const Matrix<T, ROWS, COLS>& matrix1,
                       const Matrix<T, ROWS, COLS>& matrix2) {
    return !(matrix1 == matrix2);
}

template <class T, Dimension auto ROWS, Dimension auto COLS>
__host__ __device__ inline void Transpose(Matrix<T, ROWS, COLS>& result,
                      const Matrix<T, ROWS, COLS>& matrix1) {
#ifdef USE_ISPC
    ispc::Transpose(matrix1, result, ROWS, COLS);
#else
    for (uint32_t i = 0; i < ROWS; i++) {
        for (uint32_t j = 0; j < COLS; j++) {
            result[j * ROWS + i] = matrix1[i * COLS + j];
        }
    }
#endif
}

template <class T, Dimension auto N>
__host__ __device__ inline T Trace(const Matrix<T, N, N>& matrix) {
    T result = (T)0;

    for (Dimension auto i = 0; i < N; i++) {
        result += matrix[i][i];
    }

    return result;
}

template <class T, Dimension auto ROWS, Dimension auto COLS>
__host__ __device__ inline void DotProduct3(Vector<T, 3>& result, Vector<T, 3>& source,
                        const Matrix<T, ROWS, COLS>& matrix) {
    static_assert(
        ROWS >= 3,
        "[Error] Only 3x3 and above matrix can be passed to this method!");
    static_assert(
        COLS >= 3,
        "[Error] Only 3x3 and above matrix can be passed to this method!");
    Vector<T, 3> basis[3] = {
        {matrix[0][0], matrix[1][0], matrix[2][0]},
        {matrix[0][1], matrix[1][1], matrix[2][1]},
        {matrix[0][2], matrix[1][2], matrix[2][2]},
    };
    DotProduct(result[0], source, basis[0]);
    DotProduct(result[1], source, basis[1]);
    DotProduct(result[2], source, basis[2]);
}

template <class T, Dimension auto ROWS, Dimension auto COLS>
__host__ __device__ inline void GetOrigin(Vector<T, 3>& result,
                      const Matrix<T, ROWS, COLS>& matrix) {
    static_assert(
        ROWS >= 3,
        "[Error] Only 3x3 and above matrix can be passed to this method!");
    static_assert(
        COLS >= 3,
        "[Error] Only 3x3 and above matrix can be passed to this method!");
    result = {matrix[3][0], matrix[3][1], matrix[3][2]};
}

inline void MatrixRotationYawPitchRoll(Matrix4X4f& matrix, const float yaw,
                                       const float pitch, const float roll) {
    float cYaw, cPitch, cRoll, sYaw, sPitch, sRoll;

    // Get the cosine and sin of the yaw, pitch, and roll.
    cYaw = std::cos(yaw);
    cPitch = std::cos(pitch);
    cRoll = std::cos(roll);

    sYaw = std::sin(yaw);
    sPitch = std::sin(pitch);
    sRoll = std::sin(roll);

    // Calculate the yaw, pitch, roll rotation matrix.
    matrix = {{{(cRoll * cYaw) + (sRoll * sPitch * sYaw), (sRoll * cPitch),
                (cRoll * -sYaw) + (sRoll * sPitch * cYaw), 0.0f},
               {(-sRoll * cYaw) + (cRoll * sPitch * sYaw), (cRoll * cPitch),
                (sRoll * sYaw) + (cRoll * sPitch * cYaw), 0.0f},
               {(cPitch * sYaw), -sPitch, (cPitch * cYaw), 0.0f},
               {0.0f, 0.0f, 0.0f, 1.0f}}};
}

__host__ __device__ inline void TransformCoord(Vector3f& vector, const Matrix4X4f& matrix) {
    Vector4f tmp;
#ifdef USE_ISPC
    ispc::Transform(tmp, matrix);
#else
    for (int index = 0; index < 4; index++) {
        tmp[index] =
            (vector[0] * matrix[0][index]) + (vector[1] * matrix[1][index]) +
            (vector[2] * matrix[2][index]) + (1.0f * matrix[2][index]);
    }
#endif
    vector = tmp;
}

__host__ __device__ inline void Transform(Vector4f& vector, const Matrix4X4f& matrix) {
    Vector4f tmp;
#ifdef USE_ISPC
    ispc::Transform(vector, matrix);
#else
    for (int index = 0; index < 4; index++) {
        tmp[index] =
            (vector[0] * matrix[0][index]) + (vector[1] * matrix[1][index]) +
            (vector[2] * matrix[2][index]) + (vector[3] * matrix[2][index]);
    }
#endif
    vector = tmp;
}

template <class T, Dimension auto ROWS, Dimension auto COLS>
inline void ExchangeYandZ(Matrix<T, ROWS, COLS>& matrix) {
#ifdef USE_ISPC
    ispc::MatrixExchangeYandZ(matrix, ROWS, COLS);
#else
    for (int32_t row_index = 0; row_index < ROWS; row_index++) {
        std::swap<T>(matrix[row_index][1], matrix[row_index][2]);
    }
#endif
}

inline void BuildViewLHMatrix(Matrix4X4f& result, const Vector3f position,
                              const Vector3f lookAt, const Vector3f up) {
    Vector3f zAxis, xAxis, yAxis;
    float result1, result2, result3;

    zAxis = lookAt - position;
    Normalize(zAxis);

    CrossProduct(xAxis, up, zAxis);
    Normalize(xAxis);

    CrossProduct(yAxis, zAxis, xAxis);

    DotProduct(result1, xAxis, position);
    result1 = -result1;

    DotProduct(result2, yAxis, position);
    result2 = -result2;

    DotProduct(result3, zAxis, position);
    result3 = -result3;

    // Set the computed values in the view matrix.
    Matrix4X4f tmp = {{{xAxis[0], yAxis[0], zAxis[0], 0.0f},
                       {xAxis[1], yAxis[1], zAxis[1], 0.0f},
                       {xAxis[2], yAxis[2], zAxis[2], 0.0f},
                       {result1, result2, result3, 1.0f}}};

    result = tmp;
}

inline void BuildViewRHMatrix(Matrix4X4f& result, const Vector3f position,
                              const Vector3f lookAt, const Vector3f up) {
    Vector3f zAxis, xAxis, yAxis;
    float result1, result2, result3;

    zAxis = position - lookAt;
    Normalize(zAxis);

    CrossProduct(xAxis, up, zAxis);
    Normalize(xAxis);

    CrossProduct(yAxis, zAxis, xAxis);

    DotProduct(result1, xAxis, position);
    result1 = -result1;

    DotProduct(result2, yAxis, position);
    result2 = -result2;

    DotProduct(result3, zAxis, position);
    result3 = -result3;

    // Set the computed values in the view matrix.
    Matrix4X4f tmp = {{{xAxis[0], yAxis[0], zAxis[0], 0.0f},
                       {xAxis[1], yAxis[1], zAxis[1], 0.0f},
                       {xAxis[2], yAxis[2], zAxis[2], 0.0f},
                       {result1, result2, result3, 1.0f}}};

    result = tmp;
}

template <class T>
__host__ __device__ inline auto BuildIdentityMatrix3X3() {
    return Matrix<T, 3, 3>({{
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0}
    }});
}

template <class T>
__host__ __device__ inline auto BuildIdentityMatrix4X4() {
    return Matrix<T, 4, 4> ({{
        {1.0, 0.0, 0.0, 0.0},
        {0.0, 1.0, 0.0, 0.0},
        {0.0, 0.0, 1.0, 0.0},
        {0.0, 0.0, 0.0, 1.0}
    }});
}

template <class T, Dimension auto N>
__host__ __device__ inline void BuildIdentityMatrix(Matrix<T, N, N>& matrix) {
#ifdef USE_ISPC
    ispc::BuildIdentityMatrix(matrix, N);
#else
    memset(&matrix[0][0], 0x00, sizeof(float) * N);

    for (int32_t i = 0; i < N; i++) {
        matrix[i][i] = 1.0f;
    }
#endif
}

inline void BuildOrthographicRHMatrix(Matrix4X4f& matrix, const float left,
                                      const float right, const float top,
                                      const float bottom,
                                      const float near_plane,
                                      const float far_plane) {
    const float width = right - left;
    const float height = top - bottom;
    const float depth = far_plane - near_plane;

    matrix = {{{2.0f / width, 0.0f, 0.0f, 0.0f},
               {0.0f, 2.0f / height, 0.0f, 0.0f},
               {0.0f, 0.0f, -1.0f / depth, 0.0f},
               {-(right + left) / width, -(top + bottom) / height,
                -0.5f * (far_plane + near_plane) / depth + 0.5f, 1.0f}}};
}

inline void BuildOpenglOrthographicRHMatrix(Matrix4X4f& matrix,
                                            const float left, const float right,
                                            const float top, const float bottom,
                                            const float near_plane,
                                            const float far_plane) {
    const float width = right - left;
    const float height = top - bottom;
    const float depth = far_plane - near_plane;

    matrix = {{{2.0f / width, 0.0f, 0.0f, 0.0f},
               {0.0f, 2.0f / height, 0.0f, 0.0f},
               {0.0f, 0.0f, -2.0f / depth, 0.0f},
               {-(right + left) / width, -(top + bottom) / height,
                -(far_plane + near_plane) / depth, 1.0f}}};
}

inline void BuildPerspectiveFovLHMatrix(Matrix4X4f& matrix,
                                        const float fieldOfView,
                                        const float screenAspect,
                                        const float screenNear,
                                        const float screenDepth) {
    Matrix4X4f perspective = {
        {{1.0f / (screenAspect * std::tan(fieldOfView * 0.5f)), 0.0f, 0.0f,
          0.0f},
         {0.0f, 1.0f / std::tan(fieldOfView * 0.5f), 0.0f, 0.0f},
         {0.0f, 0.0f, screenDepth / (screenDepth - screenNear), 1.0f},
         {0.0f, 0.0f, (-screenNear * screenDepth) / (screenDepth - screenNear),
          0.0f}}};

    matrix = perspective;
}

inline void BuildPerspectiveFovRHMatrix(Matrix4X4f& matrix,
                                        const float fieldOfView,
                                        const float screenAspect,
                                        const float screenNear,
                                        const float screenDepth) {
    Matrix4X4f perspective = {
        {{1.0f / (screenAspect * std::tan(fieldOfView * 0.5f)), 0.0f, 0.0f,
          0.0f},
         {0.0f, 1.0f / std::tan(fieldOfView * 0.5f), 0.0f, 0.0f},
         {0.0f, 0.0f, screenDepth / (screenNear - screenDepth), -1.0f},
         {0.0f, 0.0f, (-screenNear * screenDepth) / (screenDepth - screenNear),
          0.0f}}};

    matrix = perspective;
}

inline void BuildOpenglPerspectiveFovRHMatrix(Matrix4X4f& matrix,
                                              const float fieldOfView,
                                              const float screenAspect,
                                              const float screenNear,
                                              const float screenDepth) {
    Matrix4X4f perspective = {
        {{1.0f / (screenAspect * std::tan(fieldOfView * 0.5f)), 0.0f, 0.0f,
          0.0f},
         {0.0f, 1.0f / std::tan(fieldOfView * 0.5f), 0.0f, 0.0f},
         {0.0f, 0.0f, (screenNear + screenDepth) / (screenNear - screenDepth),
          -1.0f},
         {0.0f, 0.0f,
          (-2.0f * screenNear * screenDepth) / (screenDepth - screenNear),
          0.0f}}};

    matrix = perspective;
}

inline void MatrixTranslation(Matrix4X4f& matrix, const float x, const float y,
                              const float z) {
    Matrix4X4f translation = {{{1.0f, 0.0f, 0.0f, 0.0f},
                               {0.0f, 1.0f, 0.0f, 0.0f},
                               {0.0f, 0.0f, 1.0f, 0.0f},
                               {x, y, z, 1.0f}}};

    matrix = translation;
}

inline void MatrixTranslation(Matrix4X4f& matrix, const Vector3f& v) {
    MatrixTranslation(matrix, v[0], v[1], v[2]);
}

inline void MatrixTranslation(Matrix4X4f& matrix, const Vector4f& v) {
    assert(v[3]);
    MatrixTranslation(matrix, v[0] / v[3], v[1] / v[3], v[2] / v[3]);
}

inline void MatrixRotationX(Matrix4X4f& matrix, const float angle) {
    const float c = std::cos(angle), s = std::sin(angle);

    matrix = {{
        {1.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, c, s, 0.0f},
        {0.0f, -s, c, 0.0f},
        {0.0f, 0.0f, 0.0f, 1.0f},
    }};
}

inline void MatrixRotationY(Matrix4X4f& matrix, const float angle) {
    const float c = std::cos(angle), s = std::sin(angle);

    matrix = {{
        {c, 0.0f, -s, 0.0f},
        {0.0f, 1.0f, 0.0f, 0.0f},
        {s, 0.0f, c, 0.0f},
        {0.0f, 0.0f, 0.0f, 1.0f},
    }};
}

inline void MatrixRotationZ(Matrix4X4f& matrix, const float angle) {
    const float c = std::cos(angle), s = std::sin(angle);

    matrix = {{{c, s, 0.0f, 0.0f},
               {-s, c, 0.0f, 0.0f},
               {0.0f, 0.0f, 1.0f, 0.0f},
               {0.0f, 0.0f, 0.0f, 1.0f}}};
}

inline void MatrixRotationAxis(Matrix4X4f& matrix, const Vector3f& axis,
                               const float angle) {
    float c = std::cos(angle), s = std::sin(angle), one_minus_c = 1.0f - c;

    Matrix4X4f rotation = {
        {{c + axis[0] * axis[0] * one_minus_c,
          axis[0] * axis[1] * one_minus_c + axis[2] * s,
          axis[0] * axis[2] * one_minus_c - axis[1] * s, 0.0f},
         {axis[0] * axis[1] * one_minus_c - axis[2] * s,
          c + axis[1] * axis[1] * one_minus_c,
          axis[1] * axis[2] * one_minus_c + axis[0] * s, 0.0f},
         {axis[0] * axis[2] * one_minus_c + axis[1] * s,
          axis[1] * axis[2] * one_minus_c - axis[0] * s,
          c + axis[2] * axis[2] * one_minus_c, 0.0f},
         {0.0f, 0.0f, 0.0f, 1.0f}}};

    matrix = rotation;
}

template <class T>
inline void MatrixRotationQuaternion(Matrix4X4f& matrix, Quaternion<T> q) {
    Matrix4X4f rotation = {
        {{1.0f - 2.0f * q[1] * q[1] - 2.0f * q[2] * q[2],
          2.0f * q[0] * q[1] + 2.0f * q[3] * q[2],
          2.0f * q[0] * q[2] - 2.0f * q[3] * q[1], 0.0f},
         {2.0f * q[0] * q[1] - 2.0f * q[3] * q[2],
          1.0f - 2.0f * q[0] * q[0] - 2.0f * q[2] * q[2],
          2.0f * q[1] * q[2] + 2.0f * q[3] * q[0], 0.0f},
         {2.0f * q[0] * q[2] + 2.0f * q[3] * q[1],
          2.0f * q[1] * q[2] - 2.0f * q[3] * q[0],
          1.0f - 2.0f * q[0] * q[0] - 2.0f * q[1] * q[1], 0.0f},
         {0.0f, 0.0f, 0.0f, 1.0f}}};

    matrix = rotation;
}

inline void MatrixScale(Matrix4X4f& matrix, const float x, const float y,
                        const float z) {
    Matrix4X4f scale = {{
        {x, 0.0f, 0.0f, 0.0f},
        {0.0f, y, 0.0f, 0.0f},
        {0.0f, 0.0f, z, 0.0f},
        {0.0f, 0.0f, 0.0f, 1.0f},
    }};

    matrix = scale;
}

inline void MatrixScale(Matrix4X4f& matrix, const Vector3f& v) {
    MatrixScale(matrix, v[0], v[1], v[2]);
}

inline void MatrixScale(Matrix4X4f& matrix, const Vector4f& v) {
    assert(v[3]);
    MatrixScale(matrix, v[0] / v[3], v[1] / v[3], v[2] / v[3]);
}

__host__ __device__ static float invf3X3(int i, int j, const float matrix[9]) {
    int pre_i = ((i == 0) ? 2 : i - 1);
    int next_i = ((i + 1 == 3) ? 0 : i + 1);
    int pre_j = ((j == 0) ? 2 : j - 1);
    int next_j = ((j + 1 == 3) ? 0 : j + 1);

#define e(a, b) matrix[(a)*3 + (b)]
    float inv = +e(next_i, next_j) * e(pre_i, pre_j) -
                e(next_i, pre_j) * e(pre_i, next_j);

    return inv;
#undef e
}

__host__ __device__ static float invf4X4(int i, int j, const float matrix[16]) {
    int pre_i = ((i == 0) ? 3 : i - 1);
    int next_i = ((i + 1 == 4) ? 0 : i + 1);
    int next_next_i = ((i + 2 >= 4) ? i - 2 : i + 2);
    int pre_j = ((j == 0) ? 3 : j - 1);
    int next_j = ((j + 1 == 4) ? 0 : j + 1);
    int next_next_j = ((j + 2 >= 4) ? j - 2 : j + 2);
    int o = i - j < 0 ? j - i : i - j;

#define e(a, b) matrix[(a)*4 + (b)]
    float inv =
        +e(next_i, next_j) * e(next_next_i, next_next_j) * e(pre_i, pre_j) +
        e(next_i, next_next_j) * e(next_next_i, pre_j) * e(pre_i, next_j) +
        e(next_i, pre_j) * e(next_next_i, next_j) * e(pre_i, next_next_j) -
        e(next_i, next_j) * e(next_next_i, pre_j) * e(pre_i, next_next_j) -
        e(next_i, next_next_j) * e(next_next_i, next_j) * e(pre_i, pre_j) -
        e(next_i, pre_j) * e(next_next_i, next_next_j) * e(pre_i, next_j);

    return (o & 0x1) ? -inv : inv;
#undef e
}

inline bool InverseMatrix3X3f(Matrix3X3f& matrix) {
#ifdef USE_ISPC
    return ispc::InverseMatrix3X3f(matrix);
#else
    float inv[9];
    double D = 0;

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            inv[j * 3 + i] = invf3X3(i, j, matrix);
        }
    }

    for (int k = 0; k < 3; k++) {
        float s = ((float *)matrix)[k] * inv[k * 3];
        D += s;
    }

    if (D == 0) return false;

    D = 1.0 / D;

    for (int i = 0; i < 9; i++) {
        ((float *)matrix)[i] = static_cast<float>(inv[i] * D);
    }

    return true;
#endif
}

inline bool InverseMatrix4X4f(Matrix4X4f& matrix) {
#ifdef USE_ISPC
    return ispc::InverseMatrix4X4f(matrix);
#else
    float inv[16];
    double D = 0;

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            inv[j * 4 + i] = invf4X4(i, j, matrix);
        }
    }

    for (int k = 0; k < 4; k++) {
        float s = ((float *)matrix)[k] * inv[k * 4];
        D += s;
    }

    if (D == 0) return false;

    D = 1.0 / D;

    for (int i = 0; i < 16; i++) {
        ((float *)matrix)[i] = static_cast<float>(inv[i] * D);
    }

#endif
}

__host__ __device__ inline float normalizing_scale_factor(float a) {
    return static_cast<float>((a == 0) ? 1.0f / sqrt(2.0f) : 1.0f);
}

constexpr float PI_over_sixteen = PI / 16.0;
constexpr float one_over_four = 1.0f / 4.0f;

inline Matrix8X8f DCT8X8(const Matrix8X8f& matrix) {
    Matrix8X8f result;
#ifdef USE_ISPC
    ispc::DCT8X8(matrix, result);
#else
    for (int u = 0; u < 8; u++) {
        for (int v = 0; v < 8; v++) {
            result[u][v] = 0;

            for (int x = 0; x < 8; x++) {
                for (int y = 0; y < 8; y++) {
                    float scale_factor =
                        one_over_four *
                        normalizing_scale_factor(static_cast<float>(u)) *
                        normalizing_scale_factor(static_cast<float>(v));
                    auto normial = static_cast<float>(
                        matrix[x][y] *
                        cos((2.0f * x + 1.0f) * u * PI_over_sixteen) *
                        cos((2.0f * y + 1.0f) * v * PI_over_sixteen));

                    result[u][v] += scale_factor * normial;
                }
            }
        }
    }
#endif
    return result;
}

inline Matrix8X8f IDCT8X8(const Matrix8X8f& matrix) {
    Matrix8X8f result;
#ifdef USE_ISPC
    ispc::IDCT8X8(matrix, result);
#else
    const float PI_over_sixteen = PI / 16.0;
    const float one_over_four = 1.0f / 4.0f;

    for (int x = 0; x < 8; x++) {
        for (int y = 0; y < 8; y++) {
            result[x][y] = 0;

            for (int u = 0; u < 8; u++) {
                for (int v = 0; v < 8; v++) {
                    float scale_factor =
                        one_over_four *
                        normalizing_scale_factor(static_cast<float>(u)) *
                        normalizing_scale_factor(static_cast<float>(v));
                    auto normial = static_cast<float>(
                        matrix[u][v] *
                        cos((2.0f * x + 1.0f) * u * PI_over_sixteen) *
                        cos((2.0f * y + 1.0f) * v * PI_over_sixteen));

                    result[x][y] += scale_factor * normial;
                }
            }
        }
    }
#endif
    return result;
}

template <class T>
using Point2D = Vector<T, 2>;
using Point2Df = Point2D<float>;

template <class T>
using Point2DPtr = std::shared_ptr<Point2D<T>>;
using Point2DPtrf = Point2DPtr<float>;

template <class T>
using Point2DList = std::vector<Point2DPtr<T>>;
using Point2DListf = Point2DList<float>;

template <class T>
using Point = Vector<T, 3>;
using Pointf = Point<float>;

template <class T>
using PointPtr = std::shared_ptr<Point<T>>;
using PointPtrf = PointPtr<float>;

template <class T>
using PointSet = std::unordered_set<PointPtr<T>>;
using PointSetf = PointSet<float>;

template <class T>
using PointList = std::vector<PointPtr<T>>;
using PointListf = PointList<float>;

template <class T>
using Edge = std::pair<PointPtr<T>, PointPtr<T>>;
template <class T>
inline bool operator==(const Edge<T>& a, const Edge<T>& b) {
    return (a.first == b.first && a.second == b.second) ||
           (a.first == b.second && a.second == b.first);
};

template <class T>
using EdgePtr = std::shared_ptr<Edge<T>>;

template <class T>
inline bool operator==(const EdgePtr<T>& a, const EdgePtr<T>& b) {
    return (a->first == b->first && a->second == b->second) ||
           (a->first == b->second && a->second == b->first);
};

template <class T>
using EdgeSet = std::unordered_set<EdgePtr<T>>;

template <class T>
using EdgeList = std::vector<EdgePtr<T>>;

template <class T>
struct Face {
    EdgeList<T> Edges;
    Vector3f Normal;
    [[nodiscard]] PointList<T> GetVertices() const {
        PointList<T> vertices;
        for (const auto& edge : Edges) {
            vertices.push_back(edge->first);
        }

        return vertices;
    }
};

template <class T>
using FacePtr = std::shared_ptr<Face<T>>;

template <class T>
using FaceSet = std::unordered_set<FacePtr<T>>;

template <class T>
using FaceList = std::vector<FacePtr<T>>;

template <class T>
inline float PointToPlaneDistance(const PointList<T>& vertices,
                                  const Point<T>& point) {
    Vector3f normal;
    float distance;
    assert(vertices.size() > 2);
    auto A = vertices[0];
    auto B = vertices[1];
    auto C = vertices[2];
    CrossProduct(normal, *B - *A, *C - *A);
    Normalize(normal);
    DotProduct(distance, normal, point - *A);

    return distance;
}

template <class T>
inline bool isPointAbovePlane(const PointList<T>& vertices,
                              const Point<T>& point) {
    return PointToPlaneDistance(vertices, point) > 0;
}

template <class T>
inline bool isPointAbovePlane(const FacePtr<T>& pface, const Point<T>& point) {
    assert(pface->Edges.size() > 2);
    PointList<T> vertices = {pface->Edges[0]->first, pface->Edges[1]->first,
                             pface->Edges[2]->first};
    return isPointAbovePlane(vertices, point);
}

template <class T>
inline T degrees_to_radians(T degrees) {
    return degrees * PI / 180.0;
}

template <class T>
inline T clamp(T x, T min, T max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

}  // namespace My
