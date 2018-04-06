#pragma once
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <iomanip>
#include <limits>
#include <cmath>
#include <memory>
#include <set>
#include <unordered_set>
#include <vector>
#include "CrossProduct.h"
#include "MulByElement.h"
#include "Normalize.h"
#include "Transform.h"
#include "Transpose.h"
#include "AddByElement.h"
#include "SubByElement.h"
#include "MatrixExchangeYandZ.h"
#include "InverseMatrix4X4f.h"
#include "DCT.h"
#include "Absolute.h"
#include "Pow.h"
#include "DivByElement.h"

#ifndef PI
#define PI 3.14159265358979323846f
#endif

#ifndef TWO_PI
#define TWO_PI 3.14159265358979323846f * 2.0f
#endif

namespace My {
    typedef float Scalar;

#ifdef max
    #undef max
#endif
#ifdef min
    #undef min
#endif

    template<typename T>
        constexpr float normalize(T value) {
            return value < 0
                ? -static_cast<float>(value) / std::numeric_limits<T>::min()
                :  static_cast<float>(value) / std::numeric_limits<T>::max()
                ;
        }

    template <typename T, int N>
    struct Vector
    {
        T data[N];

        Vector() = default;
        Vector(const T val)
        {
            for (int i = 0; i < N; i++)
            {
                data[i] = val;
            }
        }

        Vector(std::initializer_list<const T> list)
        {
            size_t i = 0;
            for (auto val : list)
            {
                data[i++] = val;
            }
        }

        operator T*() { return reinterpret_cast<T*>(this); };

        operator const T*() const { return reinterpret_cast<const T*>(this); }

        void Set(const T val)
        {
            for (int i = 0; i < N; i++)
            {
                data[i] = val;
            }
        }

        void Set(const T* pval)
        {
            memcpy(data, pval, sizeof(T) * N);
        }

        void Set(std::initializer_list<const T> list)
        {
            size_t i = 0;
            for (auto val : list)
            {
                data[i++] = val;
            }
        }

        Vector& operator=(const T* pf) 
        { 
            Set(pf);
            return *this;
        };

        Vector& operator=(const T f) 
        { 
            Set(f);
            return *this;
        };

        Vector& operator=(const Vector& v) 
        { 
            memcpy(this, &v, sizeof(v));
            return *this;
        };
    };

    typedef Vector<float, 2> Vector2f;

    typedef Vector<float, 3> Vector3f;
    typedef Vector<double, 3> Vector3;
    typedef Vector<int16_t, 3> Vector3i16;
    typedef Vector<int32_t, 3> Vector3i32;

    typedef Vector<float, 4> Vector4f;
    typedef Vector<uint8_t, 4> R8G8B8A8Unorm;
    typedef Vector<uint8_t, 4> Vector4i;

    template<typename T>
    class Quaternion : public Vector<T, 4>
    {
    public:
        using Vector<T, 4>::Vector;
	Quaternion() = default;
        Quaternion(const Vector<T, 4> rhs)
        {
            std::memcpy(this, &rhs, sizeof(Quaternion));    
        }
    };

    template <typename T, int N>
    std::ostream& operator<<(std::ostream& out, Vector<T, N> vector)
    {
        out.precision(4);
        out.setf(std::ios::fixed);
        out << "( ";
        for (uint32_t i = 0; i < N; i++) {
                out << vector.data[i] << ((i == N - 1)? ' ' : ',');
        }
        out << ")" << std::endl;

        return out;
    }

    template <typename T, int N>
    void VectorAdd(Vector<T, N>& result, const Vector<T, N>& vec1, const Vector<T, N>& vec2)
    {
        ispc::AddByElement(vec1, vec2, result, N);
    }

    template <typename T, int N>
    Vector<T, N> operator+(const Vector<T, N>& vec1, const Vector<T, N>& vec2)
    {
        Vector<T, N> result;
        VectorAdd(result, vec1, vec2);

        return result;
    }

    template <typename T, int N>
    Vector<T, N> operator+(const Vector<T, N>& vec, const T scalar)
    {
        Vector<T, N> result(scalar);
        VectorAdd(result, vec, result);

        return result;
    }

    template <typename T, int N>
    void VectorSub(Vector<T, N>& result, const Vector<T, N>& vec1, const Vector<T, N>& vec2)
    {
        ispc::SubByElement(vec1, vec2, result, N);
    }

    template <typename T, int N>
    Vector<T, N> operator-(const Vector<T, N>& vec1, const Vector<T, N>& vec2)
    {
        Vector<T, N> result;
        VectorSub(result, vec1, vec2);

        return result;
    }

    template <typename T, int N>
    Vector<T, N> operator-(const Vector<T, N>& vec, const T scalar)
    {
        Vector<T, N> result(scalar);
        VectorSub(result, vec, result);

        return result;
    }

    template <typename T, int N>
    inline void CrossProduct(Vector<T, N>& result, const Vector<T, N>& vec1, const Vector<T, N>& vec2)
    {
        ispc::CrossProduct(vec1, vec2, result);
    }

    template <typename T>
    inline void DotProduct(T& result, const T* a, const T* b, const size_t count)
    {
        T* _result = new T[count];

        result = static_cast<T>(0);

        ispc::MulByElement(a, b, _result, count);
        for (size_t i = 0; i < count; i++) {
            result += _result[i];
        }

        delete[] _result;
    }

    template <typename T, int N>
    inline void DotProduct(T& result, const Vector<T, N>& vec1, const Vector<T, N>& vec2)
    {
        DotProduct(result, static_cast<const T*>(vec1), static_cast<const T*>(vec2), N);
    }

    template <typename T, int N>
    inline void MulByElement(Vector<T, N>& result, const Vector<T, N>& a, const Vector<T, N>& b)
    {
        ispc::MulByElement(a, b, result, N);
    }

    template <typename T, int N>
    inline void MulByElement(Vector<T, N>& result, const Vector<T, N>& a, const T scalar)
    {
        Vector<T, N> v(scalar);
        ispc::MulByElement(a, v, result, N);
    }

    template <typename T, int N>
    Vector<T, N> operator*(const Vector<T, N>& vec, const Scalar scalar)
    {
        Vector<T, N> result;
        MulByElement(result, vec, scalar);

        return result;
    }

    template <typename T, int N>
    Vector<T, N> operator*(const Scalar scalar, const Vector<T, N>& vec)
    {
        Vector<T, N> result;
        MulByElement(result, vec, scalar);

        return result;
    }

    template <typename T, int N>
    Vector<T, N> operator*(const Vector<T, N>& vec1, const Vector<T, N>& vec2)
    {
        Vector<T, N> result;
        MulByElement(result, vec1, vec2);

        return result;
    }

    template <typename T, int N>
    inline void DivByElement(Vector<T, N>& result, const Vector<T, N>& a, const Vector<T, N>& b)
    {
        ispc::DivByElement(a, b, result, N);
    }

    template <typename T, int N>
    inline void DivByElement(Vector<T, N>& result, const Vector<T, N>& a, const T scalar)
    {
        Vector<T, N> v(scalar);
        ispc::DivByElement(a, v, result, N);
    }

    template <typename T, int N>
    Vector<T, N> operator/(const Vector<T, N>& vec, const Scalar scalar)
    {
        Vector<T, N> result;
        DivByElement(result, vec, scalar);

        return result;
    }

    template <typename T, int N>
    Vector<T, N> operator/=(const Vector<T, N>& vec, const Scalar scalar)
    {
        return vec / scalar;
    }

    template <typename T, int N>
    Vector<T, N> operator/(const Scalar scalar, const Vector<T, N>& vec)
    {
        Vector<T, N> result;
        DivByElement(result, vec, scalar);

        return result;
    }

    template <typename T, int N>
    Vector<T, N> operator/(const Vector<T, N>& vec1, const Vector<T, N>& vec2)
    {
        Vector<T, N> result;
        DivByElement(result, vec1, vec2);

        return result;
    }

    template <typename T, int N>
    Vector<T, N> operator/=(const Vector<T, N>& vec1, const Vector<T, N>& vec2)
    {
        return vec1 / vec2;
    }

    template <typename T>
    inline T pow(const T base, const Scalar exponent)
    {
        return std::pow(base, exponent);
    }

    template <typename T, int N>
    Vector<T, N> pow(const Vector<T, N>& vec, const Scalar exponent)
    {
        Vector<T, N> result;
        ispc::Pow(vec, N, exponent, result);
        return result;
    }

    template <typename T>
    inline T abs(const T data)
    {
        return std::abs(data);
    }

    template <typename T, int N>
    Vector<T, N> abs(const Vector<T, N>& vec)
    {
        Vector<T, N> result;
        ispc::Absolute(result, vec, N);
        return result;
    }

    template <typename T, int N>
    inline T Length(const Vector<T, N>& vec)
    {
        T result;
        DotProduct(result, vec, vec);
        return static_cast<T>(sqrt(result));
    }

    template <typename T, int N>
    inline bool operator>=(Vector<T, N>&vec, Scalar scalar)
    {
        return Length(vec) >= scalar;
    }

    template <typename T, int N>
    inline bool operator>(Vector<T, N>&vec, Scalar scalar)
    {
        return Length(vec) > scalar;
    }

    template <typename T, int N>
    inline bool operator<=(Vector<T, N>&vec, Scalar scalar)
    {
        return Length(vec) <= scalar;
    }

    template <typename T, int N>
    inline bool operator<(Vector<T, N>&vec, Scalar scalar)
    {
        return Length(vec) < scalar;
    }

    template <typename T, int N>
    inline void Normalize(Vector<T, N>& a)
    {
        T length;
        DotProduct(length, static_cast<T*>(a), static_cast<T*>(a), N);
        length = sqrt(length);
        ispc::Normalize(N, a, length);
    }

    // Matrix

    template <typename T, int ROWS, int COLS>
    struct Matrix
    {
        Vector<T, COLS> data[ROWS];

        Vector<T, COLS>& operator[](int row_index) {
            return data[row_index];
        }

        const Vector<T, COLS>& operator[](int row_index) const {
            return data[row_index];
        }

        operator T*() { return &data[0][0]; };
        operator const T*() const { return static_cast<const T*>(&data[0][0]); };

        Matrix& operator=(const T* _data) 
        {
            std::memcpy(data, _data, sizeof(T) * COLS * ROWS);
            return *this;
        }

        Matrix& operator=(const Matrix& rhs) 
        {
            std::memcpy(data, rhs, sizeof(Matrix));
            return *this;
        }
    };

    typedef Matrix<float, 3, 3> Matrix3X3f;
    typedef Matrix<float, 4, 4> Matrix4X4f;
    typedef Matrix<int32_t, 8, 8> Matrix8X8i;
    typedef Matrix<float, 8, 8> Matrix8X8f;

    template <typename T, int ROWS, int COLS>
    std::ostream& operator<<(std::ostream& out, Matrix<T, ROWS, COLS> matrix)
    {
        out << std::endl;
        for (int i = 0; i < ROWS; i++) {
            out << matrix[i];
        }
        out << std::endl;

        return out;
    }

    template <typename T, int ROWS, int COLS>
    void MatrixAdd(Matrix<T, ROWS, COLS>& result, const Matrix<T, ROWS, COLS>& matrix1, const Matrix<T, ROWS, COLS>& matrix2)
    {
        ispc::AddByElement(matrix1, matrix2, result, ROWS * COLS);
    }

    template <typename T, int ROWS, int COLS>
    Matrix<T, ROWS, COLS> operator+(const Matrix<T, ROWS, COLS>& matrix1, const Matrix<T, ROWS, COLS>& matrix2)
    {
        Matrix<T, ROWS, COLS> result;
        MatrixAdd(result, matrix1, matrix2);

        return result;
    }

    template <typename T, int ROWS, int COLS>
    void MatrixSub(Matrix<T, ROWS, COLS>& result, const Matrix<T, ROWS, COLS>& matrix1, const Matrix<T, ROWS, COLS>& matrix2)
    {
        ispc::SubByElement(matrix1, matrix2, result, ROWS * COLS);
    }

    template <typename T, int ROWS, int COLS>
    void MatrixMulByElement(Matrix<T, ROWS, COLS>& result, const Matrix<T, ROWS, COLS>& matrix1, const Matrix<T, ROWS, COLS>& matrix2)
    {
        ispc::MulByElement(matrix1, matrix2, result, ROWS * COLS);
    }

    template <int ROWS, int COLS>
    void MatrixMulByElementi32(Matrix<int32_t, ROWS, COLS>& result, const Matrix<int32_t, ROWS, COLS>& matrix1, const Matrix<int32_t, ROWS, COLS>& matrix2)
    {
        ispc::MulByElementi32(matrix1, matrix2, result, ROWS * COLS);
    }

    template <typename T, int ROWS, int COLS>
    Matrix<T, ROWS, COLS> operator-(const Matrix<T, ROWS, COLS>& matrix1, const Matrix<T, ROWS, COLS>& matrix2)
    {
        Matrix<T, ROWS, COLS> result;
        MatrixSub(result, matrix1, matrix2);

        return result;
    }

    template <typename T, int Da, int Db, int Dc>
    void MatrixMultiply(Matrix<T, Da, Dc>& result, const Matrix<T, Da, Db>& matrix1, const Matrix<T, Dc, Db>& matrix2)
    {
        Matrix<T, Dc, Db> matrix2_transpose;
        Transpose(matrix2_transpose, matrix2);
        for (int i = 0; i < Da; i++) {
            for (int j = 0; j < Dc; j++) {
                DotProduct(result[i][j], matrix1[i], matrix2_transpose[j]);
            }
        }

        return;
    }

    template <typename T, int ROWS, int COLS>
    Matrix<T, ROWS, COLS> operator*(const Matrix<T, ROWS, COLS>& matrix1, const Matrix<T, ROWS, COLS>& matrix2)
    {
        Matrix<T, ROWS, COLS> result;
        MatrixMultiply(result, matrix1, matrix2);

        return result;
    }

    template <typename T, int ROWS, int COLS>
    Matrix<T, ROWS, COLS> operator*(const Matrix<T, ROWS, COLS>& matrix, const Scalar scalar)
    {
        Matrix<T, ROWS, COLS> result;

        for (int i = 0; i < ROWS; i++)
        {
            result[i] = matrix[i] * scalar;
        }

        return result;
    }

    template <typename T, int ROWS1, int COLS1, int ROWS2, int COLS2>
    void Shrink(Matrix<T, ROWS1, COLS1>& matrix1, const Matrix<T, ROWS2, COLS2>& matrix2)
    {
        static_assert(ROWS1 < ROWS2, "[Error] Target matrix ROWS must smaller than source matrix ROWS!");
        static_assert(COLS1 < COLS2, "[Error] Target matrix COLS must smaller than source matrix COLS!");

        const size_t size = sizeof(T) * COLS1;
        for (int i = 0; i < ROWS1; i++)
        {
            std::memcpy(matrix1[i], matrix2[i], size);
        }
    }

    template <typename T, int ROWS, int COLS>
    void Absolute(Matrix<T, ROWS, COLS>& result, const Matrix<T, ROWS, COLS>& matrix)
    {
        ispc::Absolute(result, matrix, ROWS * COLS);
    }

    template <typename T, int ROWS, int COLS>
    inline bool AlmostZero(const Matrix<T, ROWS, COLS>& matrix)
    {
        bool result = true;
        for (int i = 0; i < ROWS; i++)
        {
            if (Length(matrix[i]) > std::numeric_limits<T>::epsilon())
            {
                result = false;
            }
        }

        return result;
    }

    template <typename T, int ROWS, int COLS>
    inline bool operator==(const Matrix<T, ROWS, COLS>& matrix1, const Matrix<T, ROWS, COLS>& matrix2)
    {
        return AlmostZero(matrix1 - matrix2);
    }

    template <typename T, int ROWS, int COLS>
    inline bool operator!=(const Matrix<T, ROWS, COLS>& matrix1, const Matrix<T, ROWS, COLS>& matrix2)
    {
        return !(matrix1 == matrix2);
    }

    template <typename T, int ROWS, int COLS>
    inline void Transpose(Matrix<T, ROWS, COLS>& result, const Matrix<T, ROWS, COLS>& matrix1)
    {
        ispc::Transpose(matrix1, result, ROWS, COLS);
    }

    template <typename T, int ROWS, int COLS>
    inline void DotProduct3(Vector<T, 3>& result, Vector<T, 3>& source, const Matrix<T, ROWS, COLS>& matrix)
    {
        static_assert(ROWS >= 3, "[Error] Only 3x3 and above matrix can be passed to this method!");
        static_assert(COLS >= 3, "[Error] Only 3x3 and above matrix can be passed to this method!");
        Vector<T, 3> basis[3] = {{matrix[0][0], matrix[1][0], matrix[2][0]}, 
                         {matrix[0][1], matrix[1][1], matrix[2][1]},
                         {matrix[0][2], matrix[1][2], matrix[2][2]},
                        };
        DotProduct(result[0], source, basis[0]);
        DotProduct(result[1], source, basis[1]);
        DotProduct(result[2], source, basis[2]);
    }

    template <typename T, int ROWS, int COLS>
    inline void GetOrigin(Vector<T, 3>& result, const Matrix<T, ROWS, COLS>& matrix)
    {
        static_assert(ROWS >= 3, "[Error] Only 3x3 and above matrix can be passed to this method!");
        static_assert(COLS >= 3, "[Error] Only 3x3 and above matrix can be passed to this method!");
        result = {matrix[3][0], matrix[3][1], matrix[3][2]}; 
    }

    inline void MatrixRotationYawPitchRoll(Matrix4X4f& matrix, const float yaw, const float pitch, const float roll)
    {
        float cYaw, cPitch, cRoll, sYaw, sPitch, sRoll;

        // Get the cosine and sin of the yaw, pitch, and roll.
        cYaw = cosf(yaw);
        cPitch = cosf(pitch);
        cRoll = cosf(roll);

        sYaw = sinf(yaw);
        sPitch = sinf(pitch);
        sRoll = sinf(roll);

        // Calculate the yaw, pitch, roll rotation matrix.
        matrix = {{
            { (cRoll * cYaw) + (sRoll * sPitch * sYaw), (sRoll * cPitch), (cRoll * -sYaw) + (sRoll * sPitch * cYaw), 0.0f },
            { (-sRoll * cYaw) + (cRoll * sPitch * sYaw), (cRoll * cPitch), (sRoll * sYaw) + (cRoll * sPitch * cYaw), 0.0f },
            { (cPitch * sYaw), -sPitch, (cPitch * cYaw), 0.0f },
            { 0.0f, 0.0f, 0.0f, 1.0f }
        }};

        return;
    }

    inline void TransformCoord(Vector3f& vector, const Matrix4X4f& matrix)
    {
		Vector4f tmp ({vector[0], vector[1], vector[2], 1.0f});
        ispc::Transform(tmp, matrix);
        memcpy(&vector, &tmp, sizeof(vector));
    }

    inline void Transform(Vector4f& vector, const Matrix4X4f& matrix)
    {
        ispc::Transform(vector, matrix);

        return;
    }

    template <typename T, int ROWS, int COLS>
    inline void ExchangeYandZ(Matrix<T,ROWS,COLS>& matrix)
    {
        ispc::MatrixExchangeYandZ(matrix, ROWS, COLS);
    }

    inline void BuildViewMatrix(Matrix4X4f& result, const Vector3f position, const Vector3f lookAt, const Vector3f up)
    {
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
        Matrix4X4f tmp = {{
            { xAxis[0], yAxis[0], zAxis[0], 0.0f },
            { xAxis[1], yAxis[1], zAxis[1], 0.0f },
            { xAxis[2], yAxis[2], zAxis[2], 0.0f },
            { result1, result2, result3, 1.0f }
        }};

        result = tmp;
    }

    inline void BuildIdentityMatrix(Matrix4X4f& matrix)
    {
        Matrix4X4f identity = {{
            { 1.0f, 0.0f, 0.0f, 0.0f},
            { 0.0f, 1.0f, 0.0f, 0.0f},
            { 0.0f, 0.0f, 1.0f, 0.0f},
            { 0.0f, 0.0f, 0.0f, 1.0f}
        }};

        matrix = identity;

        return;
    }


    inline void BuildPerspectiveFovLHMatrix(Matrix4X4f& matrix, const float fieldOfView, const float screenAspect, const float screenNear, const float screenDepth)
    {
        Matrix4X4f perspective = {{
            { 1.0f / (screenAspect * tanf(fieldOfView * 0.5f)), 0.0f, 0.0f, 0.0f },
            { 0.0f, 1.0f / tanf(fieldOfView * 0.5f), 0.0f, 0.0f },
            { 0.0f, 0.0f, screenDepth / (screenDepth - screenNear), 1.0f },
            { 0.0f, 0.0f, (-screenNear * screenDepth) / (screenDepth - screenNear), 0.0f }
        }};

        matrix = perspective;

        return;
    }

    inline void BuildPerspectiveFovRHMatrix(Matrix4X4f& matrix, const float fieldOfView, const float screenAspect, const float screenNear, const float screenDepth)
    {
        Matrix4X4f perspective = {{
            { 1.0f / (screenAspect * tanf(fieldOfView * 0.5f)), 0.0f, 0.0f, 0.0f },
            { 0.0f, 1.0f / tanf(fieldOfView * 0.5f), 0.0f, 0.0f },
            { 0.0f, 0.0f, screenDepth / (screenNear - screenDepth), -1.0f },
            { 0.0f, 0.0f, (-screenNear * screenDepth) / (screenDepth - screenNear), 0.0f }
        }};

        matrix = perspective;

        return;
    }

    inline void MatrixTranslation(Matrix4X4f& matrix, const float x, const float y, const float z)
    {
        Matrix4X4f translation = {{
            { 1.0f, 0.0f, 0.0f, 0.0f},
            { 0.0f, 1.0f, 0.0f, 0.0f},
            { 0.0f, 0.0f, 1.0f, 0.0f},
            {    x,    y,    z, 1.0f}
        }};

        matrix = translation;

        return;
    }

    inline void MatrixTranslation(Matrix4X4f& matrix, const Vector3f& v)
    {
        MatrixTranslation(matrix, v[0], v[1], v[2]);
    }

    inline void MatrixTranslation(Matrix4X4f& matrix, const Vector4f& v)
    {
        assert(v[3]);
        MatrixTranslation(matrix, v[0]/v[3], v[1]/v[3], v[2]/v[3]);
    }

    inline void MatrixRotationX(Matrix4X4f& matrix, const float angle)
    {
        float c = cosf(angle), s = sinf(angle);

        Matrix4X4f rotation = {{
            {  1.0f, 0.0f, 0.0f, 0.0f },
            {  0.0f,    c,    s, 0.0f },
            {  0.0f,   -s,    c, 0.0f },
            {  0.0f, 0.0f, 0.0f, 1.0f },
        }};

        matrix = rotation;

        return;
    }

    inline void MatrixRotationY(Matrix4X4f& matrix, const float angle)
    {
        float c = cosf(angle), s = sinf(angle);

        Matrix4X4f rotation = {{
            {    c, 0.0f,   -s, 0.0f },
            { 0.0f, 1.0f, 0.0f, 0.0f },
            {    s, 0.0f,    c, 0.0f },
            { 0.0f, 0.0f, 0.0f, 1.0f },
        }};

        matrix = rotation;

        return;
    }


    inline void MatrixRotationZ(Matrix4X4f& matrix, const float angle)
    {
        float c = cosf(angle), s = sinf(angle);

        Matrix4X4f rotation = {{
            {    c,    s, 0.0f, 0.0f },
            {   -s,    c, 0.0f, 0.0f },
            { 0.0f, 0.0f, 1.0f, 0.0f },
            { 0.0f, 0.0f, 0.0f, 1.0f }
        }};

        matrix = rotation;

        return;
    }

    inline void MatrixRotationAxis(Matrix4X4f& matrix, const Vector3f& axis, const float angle)
    {
        float c = cosf(angle), s = sinf(angle), one_minus_c = 1.0f - c;

        Matrix4X4f rotation = {{
            {   c + axis[0] * axis[0] * one_minus_c,  axis[0] * axis[1] * one_minus_c + axis[2] * s, axis[0] * axis[2] * one_minus_c - axis[1] * s, 0.0f    },
            {   axis[0] * axis[1] * one_minus_c - axis[2] * s, c + axis[1] * axis[1] * one_minus_c,  axis[1] * axis[2] * one_minus_c + axis[0] * s, 0.0f    },
            {   axis[0] * axis[2] * one_minus_c + axis[1] * s, axis[1] * axis[2] * one_minus_c - axis[0] * s, c + axis[2] * axis[2] * one_minus_c, 0.0f },
            {   0.0f,  0.0f,  0.0f,  1.0f   }
        }};

        matrix = rotation;
    }

    template<typename T>
    inline void MatrixRotationQuaternion(Matrix4X4f& matrix, Quaternion<T> q)
    {
        Matrix4X4f rotation = {{
            {   1.0f - 2.0f * q[1] * q[1] - 2.0f * q[2] * q[2],  2.0f * q[0] * q[1] + 2.0f * q[3] * q[2],   2.0f * q[0] * q[2] - 2.0f * q[3] * q[1],    0.0f    },
            {   2.0f * q[0] * q[1] - 2.0f * q[3] * q[2],    1.0f - 2.0f * q[0] * q[0] - 2.0f * q[2] * q[2], 2.0f * q[1] * q[2] + 2.0f * q[3] * q[0],    0.0f    },
            {   2.0f * q[0] * q[2] + 2.0f * q[3] * q[1],    2.0f * q[1] * q[2] - 2.0f * q[1] * q[2] - 2.0f * q[3] * q[0], 1.0f - 2.0f * q[0] * q[0] - 2.0f * q[1] * q[1], 0.0f    },
            {   0.0f,   0.0f,   0.0f,   1.0f    }
        }};

        matrix = rotation;
    }

    inline void MatrixScale(Matrix4X4f& matrix, const float x, const float y, const float z)
    {
        Matrix4X4f scale = {{
            {    x, 0.0f, 0.0f, 0.0f},
            { 0.0f,    y, 0.0f, 0.0f},
            { 0.0f, 0.0f,    z, 0.0f},
            { 0.0f, 0.0f, 0.0f, 1.0f},
        }};

        matrix = scale;

        return;
    }

    inline void MatrixScale(Matrix4X4f& matrix, const Vector3f& v)
    {
        MatrixScale(matrix, v[0], v[1], v[2]);
    }

    inline void MatrixScale(Matrix4X4f& matrix, const Vector4f& v)
    {
        assert(v[3]);
        MatrixScale(matrix, v[0]/v[3], v[1]/v[3], v[2]/v[3]);
    }

    inline bool InverseMatrix3X3f(Matrix3X3f& matrix)
    {
        return ispc::InverseMatrix3X3f(matrix);
    }

    inline bool InverseMatrix4X4f(Matrix4X4f& matrix)
    {
        return ispc::InverseMatrix4X4f(matrix);
    }

    inline Matrix8X8f DCT8X8(const Matrix8X8f& matrix)
    {
        Matrix8X8f result;
        ispc::DCT8X8(matrix, result);
        return result;
    }

    inline Matrix8X8f IDCT8X8(const Matrix8X8f& matrix)
    {
        Matrix8X8f result;
        ispc::IDCT8X8(matrix, result);
        return result;
    }

    typedef Vector<float, 3> Point;
    typedef std::shared_ptr<Point> PointPtr;
    typedef std::unordered_set<PointPtr> PointSet;
    typedef std::vector<PointPtr> PointList;
    typedef std::pair<PointPtr, PointPtr> Edge;
    inline bool operator==(const Edge& a, const Edge& b)
    {
        return (a.first == b.first && a.second == b.second) || (a.first == b.second && a.second == b.first);
    }
    typedef std::shared_ptr<Edge> EdgePtr;
    inline bool operator==(const EdgePtr& a, const EdgePtr& b)
    {
        return (a->first == b->first && a->second == b->second) || (a->first == b->second && a->second == b->first);
    }
    typedef std::unordered_set<EdgePtr> EdgeSet;
    typedef std::vector<EdgePtr> EdgeList;
    struct Face {
        EdgeList    Edges;
        Vector3f    Normal;
        PointList GetVertices() const 
        {
            PointList vertices;
            for (auto edge : Edges)
            {
                vertices.push_back(edge->first);
            }

            return vertices;
        }
    };
    typedef std::shared_ptr<Face> FacePtr;
    typedef std::unordered_set<FacePtr> FaceSet;
    typedef std::vector<FacePtr> FaceList;

    inline float PointToPlaneDistance(const PointList& vertices, const Point& point)
    {
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

    inline bool isPointAbovePlane(const PointList& vertices, const Point& point)
    {
        return PointToPlaneDistance(vertices, point) > 0;
    }

    inline bool isPointAbovePlane(const FacePtr& pface, const Point& point)
    {
        assert(pface->Edges.size() > 2);
        PointList vertices = {pface->Edges[0]->first, pface->Edges[1]->first, pface->Edges[2]->first};
        return isPointAbovePlane(vertices, point);
    }
}

