#pragma once
#include <cstdint>
#include <iostream>
#include <limits>
#include <math.h>
#include "include/CrossProduct.h"
#include "include/DotProduct.h"
#include "include/MulByElement.h"
#include "include/Normalize.h"
#include "include/Transform.h"
#include "include/Transpose.h"
#include "include/AddByElement.h"
#include "include/SubByElement.h"

#ifndef PI
#define PI 3.14159265358979323846f
#endif

#ifndef TWO_PI
#define TWO_PI 3.14159265358979323846f * 2.0f
#endif

namespace My {
    template<typename T, size_t SizeOfArray>
        constexpr size_t countof(T (&)[SizeOfArray]) { return SizeOfArray; }

    template<typename T, size_t RowSize, size_t ColSize>
        constexpr size_t countof(T (&)[RowSize][ColSize]) { return RowSize * ColSize; }

    template<typename T>
        constexpr float normalize(T value) {
            return value < 0
                ? -static_cast<float>(value) / std::numeric_limits<T>::min()
                :  static_cast<float>(value) / std::numeric_limits<T>::max()
                ;
        }

    template <template<typename> class TT, typename T, int ... Indexes>
	class swizzle {
		T v[sizeof...(Indexes)];

	public:
		
		TT<T>& operator=(const TT<T>& rhs)
		{
            int indexes[] = { Indexes... };
            for (int i = 0; i < sizeof...(Indexes); i++) {
			    v[indexes[i]] = rhs[i];
            }
			return *(TT<T>*)this;
		}
	
		operator TT<T>() const
		{
			return TT<T>( v[Indexes]... );
		}
		
	};

    template <typename T>
    struct Vector2Type
    {
        union {
            T data[2];
            struct { T x, y; };
            struct { T r, g; };
            struct { T u, v; };
		    swizzle<My::Vector2Type, T, 0, 1> xy;
		    swizzle<My::Vector2Type, T, 1, 0> yx;
        };

        Vector2Type<T>() {};
        Vector2Type<T>(const T& _v) : x(_v), y(_v) {};
        Vector2Type<T>(const T& _x, const T& _y) : x(_x), y(_y) {};

        operator T*() { return data; };
        operator const T*() const { return static_cast<const T*>(data); };
    };
    
    typedef Vector2Type<float> Vector2f;

    template <typename T>
    struct Vector3Type
    {
        union {
            T data[3];
            struct { T x, y, z; };
            struct { T r, g, b; };
		    swizzle<My::Vector2Type, T, 0, 1> xy;
		    swizzle<My::Vector2Type, T, 1, 0> yx;
		    swizzle<My::Vector2Type, T, 0, 2> xz;
		    swizzle<My::Vector2Type, T, 2, 0> zx;
		    swizzle<My::Vector2Type, T, 1, 2> yz;
		    swizzle<My::Vector2Type, T, 2, 1> zy;
		    swizzle<My::Vector3Type, T, 0, 1, 2> xyz;
		    swizzle<My::Vector3Type, T, 1, 0, 2> yxz;
		    swizzle<My::Vector3Type, T, 0, 2, 1> xzy;
		    swizzle<My::Vector3Type, T, 2, 0, 1> zxy;
		    swizzle<My::Vector3Type, T, 1, 2, 0> yzx;
		    swizzle<My::Vector3Type, T, 2, 1, 0> zyx;
        };

        Vector3Type<T>() {};
        Vector3Type<T>(const T& _v) : x(_v), y(_v), z(_v) {};
        Vector3Type<T>(const T& _x, const T& _y, const T& _z) : x(_x), y(_y), z(_z) {};

        operator T*() { return data; };
        operator const T*() const { return static_cast<const T*>(data); };
    };

    typedef Vector3Type<float> Vector3f;

    template <typename T>
    struct Vector4Type
    {
        union {
            T data[4];
            struct { T x, y, z, w; };
            struct { T r, g, b, a; };
		    swizzle<My::Vector3Type, T, 0, 2, 1> xzy;
		    swizzle<My::Vector3Type, T, 1, 0, 2> yxz;
		    swizzle<My::Vector3Type, T, 1, 2, 0> yzx;
		    swizzle<My::Vector3Type, T, 2, 0, 1> zxy;
		    swizzle<My::Vector3Type, T, 2, 1, 0> zyx;
		    swizzle<My::Vector4Type, T, 2, 1, 0, 3> bgra;
        };

        Vector4Type<T>() {};
        Vector4Type<T>(const T& _v) : x(_v), y(_v), z(_v), w(_v) {};
        Vector4Type<T>(const T& _x, const T& _y, const T& _z, const T& _w) : x(_x), y(_y), z(_z), w(_w) {};
        Vector4Type<T>(const Vector3Type<T>& v3) : x(v3.x), y(v3.y), z(v3.z), w(1.0f) {};
        Vector4Type<T>(const Vector3Type<T>& v3, const T& _w) : x(v3.x), y(v3.y), z(v3.z), w(_w) {};

        operator T*() { return data; };
        operator const T*() const { return static_cast<const T*>(data); };
    };

    typedef Vector4Type<float> Vector4f;
    typedef Vector4Type<uint8_t> R8G8B8A8Unorm;
    typedef Vector4Type<uint8_t> Vector4i;

    template <template <typename> class TT, typename T>
    std::ostream& operator<<(std::ostream& out, TT<T> vector)
    {
        out << "( ";
        for (uint32_t i = 0; i < countof(vector.data); i++) {
                out << vector.data[i] << ((i == countof(vector.data) - 1)? ' ' : ',');
        }
        out << ")\n";

        return out;
    }

    template <template<typename> class TT, typename T>
    void VectorAdd(TT<T>& result, const TT<T>& vec1, const TT<T>& vec2)
    {
        ispc::AddByElement(vec1, vec2, result, countof(result.data));
    }

    template <template<typename> class TT, typename T>
    TT<T> operator+(const TT<T>& vec1, const TT<T>& vec2)
    {
        TT<T> result;
        VectorAdd(result, vec1, vec2);

        return result;
    }

    template <template<typename> class TT, typename T>
    void VectorSub(TT<T>& result, const TT<T>& vec1, const TT<T>& vec2)
    {
        ispc::SubByElement(vec1, vec2, result, countof(result.data));
    }

    template <template<typename> class TT, typename T>
    TT<T> operator-(const TT<T>& vec1, const TT<T>& vec2)
    {
        TT<T> result;
        VectorSub(result, vec1, vec2);

        return result;
    }

    template <template <typename> class TT, typename T>
    inline void CrossProduct(TT<T>& result, const TT<T>& vec1, const TT<T>& vec2)
    {
        ispc::CrossProduct(vec1, vec2, result);
    }

    template <template <typename> class TT, typename T>
    inline void DotProduct(T& result, const TT<T>& vec1, const TT<T>& vec2)
    {
        ispc::DotProduct(vec1, vec2, &result, countof(vec1.data));
    }

    template <typename T>
    inline void MulByElement(T& result, const T& a, const T& b)
    {
        ispc::MulByElement(a, b, result, countof(result.data));
    }


    // Matrix

    template <typename T, int ROWS, int COLS>
    struct Matrix
    {
        union {
            T data[ROWS][COLS];
        };

        T* operator[](int row_index) {
            return data[row_index];
        }

        const T* operator[](int row_index) const {
            return data[row_index];
        }

        operator T*() { return &data[0][0]; };
        operator const T*() const { return static_cast<const T*>(&data[0][0]); };
    };

    typedef Matrix<float, 4, 4> Matrix4X4f;

    template <typename T, int ROWS, int COLS>
    std::ostream& operator<<(std::ostream& out, Matrix<T, ROWS, COLS> matrix)
    {
        out << std::endl;
        for (int i = 0; i < ROWS; i++) {
            for (int j = 0; j < COLS; j++) {
                out << matrix.data[i][j] << ((j == COLS - 1)? '\n' : ',');
            }
        }
        out << std::endl;

        return out;
    }

    template <typename T, int ROWS, int COLS>
    void MatrixAdd(Matrix<T, ROWS, COLS>& result, const Matrix<T, ROWS, COLS>& matrix1, const Matrix<T, ROWS, COLS>& matrix2)
    {
        ispc::AddByElement(matrix1, matrix2, result, countof(result.data));
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
        ispc::AddByElement(matrix1, matrix2, result, countof(result.data));
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
                ispc::DotProduct(matrix1[i], matrix2_transpose[j], &result[i][j], Db);
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

    template <template <typename, int, int> class TT, typename T, int ROWS, int COLS>
    inline void Transpose(TT<T, ROWS, COLS>& result, const TT<T, ROWS, COLS>& matrix1)
    {
        ispc::Transpose(matrix1, result, ROWS, COLS);
    }

    template <typename T>
    inline void Normalize(T& result)
    {
        ispc::Normalize(result, countof(result.data));
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
        Matrix4X4f tmp = {{{
            { (cRoll * cYaw) + (sRoll * sPitch * sYaw), (sRoll * cPitch), (cRoll * -sYaw) + (sRoll * sPitch * cYaw), 0.0f },
            { (-sRoll * cYaw) + (cRoll * sPitch * sYaw), (cRoll * cPitch), (sRoll * sYaw) + (cRoll * sPitch * cYaw), 0.0f },
            { (cPitch * sYaw), -sPitch, (cPitch * cYaw), 0.0f },
            { 0.0f, 0.0f, 0.0f, 1.0f }
        }}};

        matrix = tmp;

        return;
    }

    inline void TransformCoord(Vector3f& vector, const Matrix4X4f& matrix)
    {
        ispc::Transform(vector, matrix);
    }

    inline void Transform(Vector4f& vector, const Matrix4X4f& matrix)
    {
        ispc::Transform(vector, matrix);

        return;
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
        Matrix4X4f tmp = {{{
            { xAxis.x, yAxis.x, zAxis.x, 0.0f },
            { xAxis.y, yAxis.y, zAxis.y, 0.0f },
            { xAxis.z, yAxis.z, zAxis.z, 0.0f },
            { result1, result2, result3, 1.0f }
        }}};

        result = tmp;
    }

    inline void BuildIdentityMatrix(Matrix4X4f& matrix)
    {
        Matrix4X4f identity = {{{
            { 1.0f, 0.0f, 0.0f, 0.0f},
            { 0.0f, 1.0f, 0.0f, 0.0f},
            { 0.0f, 0.0f, 1.0f, 0.0f},
            { 0.0f, 0.0f, 0.0f, 1.0f}
        }}};

        matrix = identity;

        return;
    }


    inline void BuildPerspectiveFovLHMatrix(Matrix4X4f& matrix, const float fieldOfView, const float screenAspect, const float screenNear, const float screenDepth)
    {
        Matrix4X4f perspective = {{{
            { 1.0f / (screenAspect * tanf(fieldOfView * 0.5f)), 0.0f, 0.0f, 0.0f },
            { 0.0f, 1.0f / tanf(fieldOfView * 0.5f), 0.0f, 0.0f },
            { 0.0f, 0.0f, screenDepth / (screenDepth - screenNear), 1.0f },
            { 0.0f, 0.0f, (-screenNear * screenDepth) / (screenDepth - screenNear), 0.0f }
        }}};

        matrix = perspective;

        return;
    }


    inline void MatrixTranslation(Matrix4X4f& matrix, const float x, const float y, const float z)
    {
        Matrix4X4f translation = {{{
            { 1.0f, 0.0f, 0.0f, 0.0f},
            { 0.0f, 1.0f, 0.0f, 0.0f},
            { 0.0f, 0.0f, 1.0f, 0.0f},
            {    x,    y,    z, 1.0f}
        }}};

        matrix = translation;

        return;
    }

    inline void MatrixRotationX(Matrix4X4f& matrix, const float angle)
    {
        float c = cosf(angle), s = sinf(angle);

        Matrix4X4f rotation = {{{
            {  1.0f, 0.0f, 0.0f, 0.0f },
            {  0.0f,    c,    s, 0.0f },
            {  0.0f,   -s,    c, 0.0f },
            {  0.0f, 0.0f, 0.0f, 1.0f },
        }}};

        matrix = rotation;

        return;
    }

    inline void MatrixRotationY(Matrix4X4f& matrix, const float angle)
    {
        float c = cosf(angle), s = sinf(angle);

        Matrix4X4f rotation = {{{
            {    c, 0.0f,   -s, 0.0f },
            { 0.0f, 1.0f, 0.0f, 0.0f },
            {    s, 0.0f,    c, 0.0f },
            { 0.0f, 0.0f, 0.0f, 1.0f },
        }}};

        matrix = rotation;

        return;
    }


    inline void MatrixRotationZ(Matrix4X4f& matrix, const float angle)
    {
        float c = cosf(angle), s = sinf(angle);

        Matrix4X4f rotation = {{{
            {    c,    s, 0.0f, 0.0f },
            {   -s,    c, 0.0f, 0.0f },
            { 0.0f, 0.0f, 1.0f, 0.0f },
            { 0.0f, 0.0f, 0.0f, 1.0f }
        }}};

        matrix = rotation;

        return;
    }

}

