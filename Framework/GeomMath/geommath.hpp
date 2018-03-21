#pragma once
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
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

        Vector2Type<T>() {}
        Vector2Type<T>(const T& _v) : x(_v), y(_v) {}
        Vector2Type<T>(const T& _x, const T& _y) : x(_x), y(_y) {}

        operator T*() { return data; };
        operator const T*() const { return static_cast<const T*>(data); }

        void Set(const T& _v) { x = _v; y = _v; }
        void Set(const T& _x, const T& _y) { x = _x; y = _y; }
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
		    swizzle<My::Vector3Type, T, 0, 1, 2> rgb;
        };

        Vector3Type<T>() {}
        Vector3Type<T>(const T& _v) : x(_v), y(_v), z(_v) {}
        Vector3Type<T>(const T& _x, const T& _y, const T& _z) : x(_x), y(_y), z(_z) {}
        
        operator T*() { return data; };
        operator const T*() const { return static_cast<const T*>(data); };

        void Set(const T& _v) { x = _v; y = _v; z=_v; }
        void Set(const T& _x, const T& _y, const T& _z) { x = _x; y = _y; z = _z; }
    };

    typedef Vector3Type<float> Vector3f;
    typedef Vector3Type<double> Vector3;
    typedef Vector3Type<int16_t> Vector3i16;
    typedef Vector3Type<int32_t> Vector3i32;

    template <typename T>
    struct Vector4Type
    {
        union {
            T data[4];
            struct { T x, y, z, w; };
            struct { T r, g, b, a; };
		    swizzle<My::Vector3Type, T, 0, 1, 2> xyz;
		    swizzle<My::Vector3Type, T, 0, 2, 1> xzy;
		    swizzle<My::Vector3Type, T, 1, 0, 2> yxz;
		    swizzle<My::Vector3Type, T, 1, 2, 0> yzx;
		    swizzle<My::Vector3Type, T, 2, 0, 1> zxy;
		    swizzle<My::Vector3Type, T, 2, 1, 0> zyx;
		    swizzle<My::Vector3Type, T, 0, 1, 2> rgb;
		    swizzle<My::Vector4Type, T, 0, 1, 2, 3> rgba;
		    swizzle<My::Vector4Type, T, 2, 1, 0, 3> bgra;
        };

        Vector4Type<T>() {};
        Vector4Type<T>(const T& _v) : x(_v), y(_v), z(_v), w(_v) {};
        Vector4Type<T>(const T& _x, const T& _y, const T& _z, const T& _w) : x(_x), y(_y), z(_z), w(_w) {};
        Vector4Type<T>(const Vector3Type<T>& v3) : x(v3.x), y(v3.y), z(v3.z), w(1.0f) {};
        Vector4Type<T>(const Vector3Type<T>& v3, const T& _w) : x(v3.x), y(v3.y), z(v3.z), w(_w) {};

        operator T*() { return data; };
        operator const T*() const { return static_cast<const T*>(data); };

        void Set(const T& _v) { x = _v; y = _v; z=_v; w=_v; }
        void Set(const T& _x, const T& _y, const T& _z, const T& _w) { x = _x; y = _y; z = _z; w = _w; }
        
        Vector4Type& operator=(const T* f) 
        { 
            for (int32_t i = 0; i < 4; i++)
            {
                data[i] = *(f + i); 
            }
            return *this;
        };
        
    };

    typedef Vector4Type<float> Vector4f;
    typedef Vector4Type<float> Quaternion;
    typedef Vector4Type<uint8_t> R8G8B8A8Unorm;
    typedef Vector4Type<uint8_t> Vector4i;

    template <template <typename> class TT>
    std::ostream& operator<<(std::ostream& out, TT<int8_t> vector)
    {
        out << "( ";
        for (uint32_t i = 0; i < countof(vector.data); i++) {
                out << (int)vector.data[i] << ((i == countof(vector.data) - 1)? ' ' : ',');
        }
        out << ")\n";

        return out;
    }

    template <template <typename> class TT>
    std::ostream& operator<<(std::ostream& out, TT<uint8_t> vector)
    {
        out << "( ";
        for (uint32_t i = 0; i < countof(vector.data); i++) {
                out << (unsigned int)vector.data[i] << ((i == countof(vector.data) - 1)? ' ' : ',');
        }
        out << ")\n";

        return out;
    }

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
    TT<T> operator+(const TT<T>& vec, const T scalar)
    {
        TT<T> result(scalar);
        VectorAdd(result, vec, result);

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

    template <template<typename> class TT, typename T>
    TT<T> operator-(const TT<T>& vec, const T scalar)
    {
        TT<T> result(scalar);
        VectorSub(result, vec, result);

        return result;
    }

    template <template <typename> class TT, typename T>
    inline void CrossProduct(TT<T>& result, const TT<T>& vec1, const TT<T>& vec2)
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

    template <template <typename> class TT, typename T>
    inline void DotProduct(T& result, const TT<T>& vec1, const TT<T>& vec2)
    {
        DotProduct(result, static_cast<const T*>(vec1), static_cast<const T*>(vec2), countof(vec1.data));
    }

    template <typename T>
    inline void MulByElement(T& result, const T& a, const T b)
    {
        ispc::MulByElement(a, b, result, countof(result.data));
    }

    template <template <typename> class TT, typename T>
    inline void MulByElement(TT<T>& result, const TT<T>& a, const T b)
    {
        TT<T> v_b(b);
        ispc::MulByElement(a, v_b, result, countof(result.data));
    }

    template <template<typename> class TT, typename T>
    TT<T> operator*(const TT<T>& vec, const T scalar)
    {
        TT<T> result;
        MulByElement(result, vec, scalar);

        return result;
    }

    template <template <typename> class TT, typename T>
    inline T Length(const TT<T>& vec)
    {
        T result;
        DotProduct(result, vec, vec);
        return static_cast<T>(sqrt(result));
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

        Matrix& operator=(const T* _data) 
        {
            for (int i = 0; i < ROWS; i++) {
                for (int j = 0; j < COLS; j++) {
                    data[i][j] = *(_data + i * COLS + j);
                }
            }
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
        ispc::SubByElement(matrix1, matrix2, result, countof(result.data));
    }

    template <typename T, int ROWS, int COLS>
    void MatrixMulByElement(Matrix<T, ROWS, COLS>& result, const Matrix<T, ROWS, COLS>& matrix1, const Matrix<T, ROWS, COLS>& matrix2)
    {
        ispc::MulByElement(matrix1, matrix2, result, countof(result.data));
    }

    template <int ROWS, int COLS>
    void MatrixMulByElementi32(Matrix<int32_t, ROWS, COLS>& result, const Matrix<int32_t, ROWS, COLS>& matrix1, const Matrix<int32_t, ROWS, COLS>& matrix2)
    {
        ispc::MulByElementi32(matrix1, matrix2, result, countof(result.data));
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
                DotProduct(result[i][j], matrix1[i], matrix2_transpose[j], Db);
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
        ispc::Absolute(result, matrix, countof(matrix.data));
    }

    template <template <typename, int, int> class TT, typename T, int ROWS, int COLS>
    inline void Transpose(TT<T, ROWS, COLS>& result, const TT<T, ROWS, COLS>& matrix1)
    {
        ispc::Transpose(matrix1, result, ROWS, COLS);
    }

    template <template <typename, int, int> class M, typename T, int ROWS, int COLS>
    inline void DotProduct3(Vector3Type<T>& result, Vector3Type<T>& source, const M<T, ROWS, COLS>& matrix)
    {
        static_assert(ROWS >= 3, "[Error] Only 3x3 and above matrix can be passed to this method!");
        static_assert(COLS >= 3, "[Error] Only 3x3 and above matrix can be passed to this method!");
        Vector3Type<T> basis[3] = {{matrix[0][0], matrix[1][0], matrix[2][0]}, 
                         {matrix[0][1], matrix[1][1], matrix[2][1]},
                         {matrix[0][2], matrix[1][2], matrix[2][2]},
                        };
        DotProduct(result.x, source, basis[0]);
        DotProduct(result.y, source, basis[1]);
        DotProduct(result.z, source, basis[2]);
    }

    template <template <typename, int, int> class M, typename T, int ROWS, int COLS>
    inline void GetOrigin(Vector3Type<T>& result, const M<T, ROWS, COLS>& matrix)
    {
        static_assert(ROWS >= 3, "[Error] Only 3x3 and above matrix can be passed to this method!");
        static_assert(COLS >= 3, "[Error] Only 3x3 and above matrix can be passed to this method!");
        result = {matrix[3][0], matrix[3][1], matrix[3][2]}; 
    }
    template <template <typename> class TT, typename T>
    inline void Normalize(TT<T>& a)
    {
        T length;
        DotProduct(length, static_cast<T*>(a), static_cast<T*>(a), countof(a.data));
        length = sqrt(length);
        ispc::Normalize(countof(a.data), a, length);
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
		Vector4f tmp (vector, 1.0f);
        ispc::Transform(tmp, matrix);
		vector.xyz = tmp.xyz;
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

    inline void BuildPerspectiveFovRHMatrix(Matrix4X4f& matrix, const float fieldOfView, const float screenAspect, const float screenNear, const float screenDepth)
    {
        Matrix4X4f perspective = {{{
            { 1.0f / (screenAspect * tanf(fieldOfView * 0.5f)), 0.0f, 0.0f, 0.0f },
            { 0.0f, 1.0f / tanf(fieldOfView * 0.5f), 0.0f, 0.0f },
            { 0.0f, 0.0f, screenDepth / (screenNear - screenDepth), -1.0f },
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

    inline void MatrixScale(Matrix4X4f& matrix, const float x, const float y, const float z)
    {
        Matrix4X4f scale = {{{
            {    x, 0.0f, 0.0f, 0.0f},
            { 0.0f,    y, 0.0f, 0.0f},
            { 0.0f, 0.0f,    z, 0.0f},
            { 0.0f, 0.0f, 0.0f, 1.0f},
        }}};

        matrix = scale;

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

    inline void MatrixRotationAxis(Matrix4X4f& matrix, const Vector3f& axis, const float angle)
    {
        float c = cosf(angle), s = sinf(angle), one_minus_c = 1.0f - c;

        Matrix4X4f rotation = {{{
            {   c + axis.x * axis.x * one_minus_c,  axis.x * axis.y * one_minus_c + axis.z * s, axis.x * axis.z * one_minus_c - axis.y * s, 0.0f    },
            {   axis.x * axis.y * one_minus_c - axis.z * s, c + axis.y * axis.y * one_minus_c,  axis.y * axis.z * one_minus_c + axis.x * s, 0.0f    },
            {   axis.x * axis.z * one_minus_c + axis.y * s, axis.y * axis.z * one_minus_c - axis.x * s, c + axis.z * axis.z * one_minus_c, 0.0f },
            {   0.0f,  0.0f,  0.0f,  1.0f   }
        }}};

        matrix = rotation;
    }

    inline void MatrixRotationQuaternion(Matrix4X4f& matrix, Quaternion q)
    {
        Matrix4X4f rotation = {{{
            {   1.0f - 2.0f * q.y * q.y - 2.0f * q.z * q.z,  2.0f * q.x * q.y + 2.0f * q.w * q.z,   2.0f * q.x * q.z - 2.0f * q.w * q.y,    0.0f    },
            {   2.0f * q.x * q.y - 2.0f * q.w * q.z,    1.0f - 2.0f * q.x * q.x - 2.0f * q.z * q.z, 2.0f * q.y * q.z + 2.0f * q.w * q.x,    0.0f    },
            {   2.0f * q.x * q.z + 2.0f * q.w * q.y,    2.0f * q.y * q.z - 2.0f * q.y * q.z - 2.0f * q.w * q.x, 1.0f - 2.0f * q.x * q.x - 2.0f * q.y * q.y, 0.0f    },
            {   0.0f,   0.0f,   0.0f,   1.0f    }
        }}};

        matrix = rotation;
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

    typedef Vector3Type<float> Point;
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

    inline bool isPointAbovePlane(const PointList& vertices, const Point& point)
    {
        auto count = vertices.size();
        assert(count > 2);
        auto ab = *vertices[1] - *vertices[0];
        auto ac = *vertices[2] - *vertices[0];
        Vector3f normal;
        float cos_theta;
        CrossProduct(normal, ab, ac);
        auto dir = point - *vertices[0];
        DotProduct(cos_theta, normal, dir);

        return cos_theta > 0;
    }

    inline bool isPointAbovePlane(const FacePtr& pface, const Point& point)
    {
        assert(pface->Edges.size() > 2);
        PointList vertices = {pface->Edges[0]->first, pface->Edges[1]->first, pface->Edges[2]->first};
        return isPointAbovePlane(vertices, point);
    }

    inline float PointToPlaneDistance(const PointList& vertices, const PointPtr& point_ptr)
    {
        Vector3f normal;
        float distance;
        auto A = vertices[0];
        auto B = vertices[1];
        auto C = vertices[2];
        CrossProduct(normal, *B - *A, *C - *A);
        Normalize(normal);
        DotProduct(distance, normal, *point_ptr - *A);
        distance = std::abs(distance);

        return distance;
    }

}

