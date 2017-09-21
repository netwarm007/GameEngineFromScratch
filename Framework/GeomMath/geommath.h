#pragma once
#include <math.h>
#include "include/CrossProduct.h"
#include "include/DotProduct.h"
#include "include/MulByElement.h"
#include "include/Transpose.h"
#include "include/Normalize.h"

#ifndef PI
#define PI 3.14159265358979323846f
#endif

#ifndef TWO_PI
#define TWO_PI 3.14159265358979323846f * 2.0f
#endif

namespace My {
	template< typename T, int ... Indexes>
	class swizzle {
		float v[sizeof...(Indexes)];

	public:
		
		T& operator=(const T& rhs)
		{
            int indexes[] = { Indexes... };
            for (int i = 0; i < sizeof...(Indexes); i++) {
			    v[indexes[i]] = rhs[i];
            }
			return *(T*)this;
		}
	
		operator T () const
		{
			return T( v[Indexes]... );
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
		    swizzle<Vector2Type<T>, 0, 1> xy;
		    swizzle<Vector2Type<T>, 1, 0> yx;
        };

        Vector2Type<T>() {};
        Vector2Type<T>(T _v) : x(_v), y(_v) {};
        Vector2Type<T>(T _x, T _y) : x(_x), y(_y) {};
    };
    
    typedef Vector2Type<float> Vector2f;

    template <typename T>
    struct Vector3Type
    {
        union {
            T data[3];
            struct { T x, y, z; };
            struct { T r, g, b; };
		    swizzle<Vector2Type<T>, 0, 1> xy;
		    swizzle<Vector2Type<T>, 1, 0> yx;
		    swizzle<Vector2Type<T>, 0, 2> xz;
		    swizzle<Vector2Type<T>, 2, 0> zx;
		    swizzle<Vector2Type<T>, 1, 2> yz;
		    swizzle<Vector2Type<T>, 2, 1> zy;
		    swizzle<Vector3Type<T>, 0, 1, 2> xyz;
		    swizzle<Vector3Type<T>, 1, 0, 2> yxz;
		    swizzle<Vector3Type<T>, 0, 2, 1> xzy;
		    swizzle<Vector3Type<T>, 2, 0, 1> zxy;
		    swizzle<Vector3Type<T>, 1, 2, 0> yzx;
		    swizzle<Vector3Type<T>, 2, 1, 0> zyx;
        };

        Vector3Type<T>() {};
        Vector3Type<T>(T _v) : x(_v), y(_v), z(_v) {};
        Vector3Type<T>(T _x, T _y, T _z) : x(_x), y(_y), z(_z) {};
    };

    typedef Vector3Type<float> Vector3f;

    template <typename T>
    struct Vector4Type
    {
        union {
            T data[4];
            struct { T x, y, z, w; };
            struct { T r, g, b, a; };
		    swizzle<Vector3Type<T>, 0, 2, 1> xzy;
		    swizzle<Vector3Type<T>, 1, 0, 2> yxz;
		    swizzle<Vector3Type<T>, 1, 2, 0> yzx;
		    swizzle<Vector3Type<T>, 2, 0, 1> zxy;
		    swizzle<Vector3Type<T>, 2, 1, 0> zyx;
        };

        Vector4Type<T>() {};
        Vector4Type<T>(T _v) : x(_v), y(_v), z(_v), w(_v) {};
        Vector4Type<T>(T _x, T _y, T _z, T _w) : x(_x), y(_y), z(_z), w(_w) {};
    };

    typedef Vector4Type<float> Vector4f;

    template <typename T, int ROWS, int COLS>
    struct Matrix
    {
        union {
            T data[ROWS * COLS];
            Vector3Type<T> row[ROWS];
        };

        T& operator[](int index) {
            return data[index];
        }

        T operator[](int index) const {
            return data[index];
        }
    };

    typedef Matrix<float, 3, 3> Matrix3X3f;
    typedef Matrix<float, 4, 4> Matrix4X4f;

    void MatrixRotationYawPitchRoll(Matrix4X4f& matrix, const float yaw, const float pitch, const float roll)
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
        matrix[0] = (cRoll * cYaw) + (sRoll * sPitch * sYaw);
        matrix[1] = (sRoll * cPitch);
        matrix[2] = (cRoll * -sYaw) + (sRoll * sPitch * cYaw);
        
        matrix[3] = (-sRoll * cYaw) + (cRoll * sPitch * sYaw);
        matrix[4] = (cRoll * cPitch);
        matrix[5] = (sRoll * sYaw) + (cRoll * sPitch * cYaw);
        
        matrix[6] = (cPitch * sYaw);
        matrix[7] = -sPitch;
        matrix[8] = (cPitch * cYaw);

        return;
    }

    void TransformCoord(Vector3f& vector, const Matrix4X4f& matrix)
    {
        float x, y, z;


        // Transform the vector by the 3x3 matrix.
        x = (vector.x * matrix[0]) + (vector.y * matrix[3]) + (vector.z * matrix[6]);
        y = (vector.x * matrix[1]) + (vector.y * matrix[4]) + (vector.z * matrix[7]);
        z = (vector.x * matrix[2]) + (vector.y * matrix[5]) + (vector.z * matrix[8]);

        // Store the result in the reference.
        vector.x = x;
        vector.y = y;
        vector.z = z;

        return;
    }

    void BuildViewMatrix(const Vector3f position, const Vector3f lookAt, const Vector3f up, Matrix4X4f& result)
    {
        Vector3f zAxis, xAxis, yAxis;
        float length, result1, result2, result3;


        // zAxis = normal(lookAt - position)
        zAxis.x = lookAt.x - position.x;
        zAxis.y = lookAt.y - position.y;
        zAxis.z = lookAt.z - position.z;
        length = sqrtf((zAxis.x * zAxis.x) + (zAxis.y * zAxis.y) + (zAxis.z * zAxis.z));
        zAxis.x = zAxis.x / length;
        zAxis.y = zAxis.y / length;
        zAxis.z = zAxis.z / length;

        // xAxis = normal(cross(up, zAxis))
        xAxis.x = (up.y * zAxis.z) - (up.z * zAxis.y);
        xAxis.y = (up.z * zAxis.x) - (up.x * zAxis.z);
        xAxis.z = (up.x * zAxis.y) - (up.y * zAxis.x);
        length = sqrtf((xAxis.x * xAxis.x) + (xAxis.y * xAxis.y) + (xAxis.z * xAxis.z));
        xAxis.x = xAxis.x / length;
        xAxis.y = xAxis.y / length;
        xAxis.z = xAxis.z / length;

        // yAxis = cross(zAxis, xAxis)
        yAxis.x = (zAxis.y * xAxis.z) - (zAxis.z * xAxis.y);
        yAxis.y = (zAxis.z * xAxis.x) - (zAxis.x * xAxis.z);
        yAxis.z = (zAxis.x * xAxis.y) - (zAxis.y * xAxis.x);

        // -dot(xAxis, position)
        result1 = ((xAxis.x * position.x) + (xAxis.y * position.y) + (xAxis.z * position.z)) * -1.0f;

        // -dot(yaxis, eye)
        result2 = ((yAxis.x * position.x) + (yAxis.y * position.y) + (yAxis.z * position.z)) * -1.0f;

        // -dot(zaxis, eye)
        result3 = ((zAxis.x * position.x) + (zAxis.y * position.y) + (zAxis.z * position.z)) * -1.0f;

        // Set the computed values in the view matrix.
        result[0]  = xAxis.x;
        result[1]  = yAxis.x;
        result[2]  = zAxis.x;
        result[3]  = 0.0f;

        result[4]  = xAxis.y;
        result[5]  = yAxis.y;
        result[6]  = zAxis.y;
        result[7]  = 0.0f;

        result[8]  = xAxis.z;
        result[9]  = yAxis.z;
        result[10] = zAxis.z;
        result[11] = 0.0f;

        result[12] = result1;
        result[13] = result2;
        result[14] = result3;
        result[15] = 1.0f;
    }

    void BuildIdentityMatrix(Matrix4X4f& matrix)
    {
        matrix[0] = 1.0f;
        matrix[1] = 0.0f;
        matrix[2] = 0.0f;
        matrix[3] = 0.0f;

        matrix[4] = 0.0f;
        matrix[5] = 1.0f;
        matrix[6] = 0.0f;
        matrix[7] = 0.0f;

        matrix[8] = 0.0f;
        matrix[9] = 0.0f;
        matrix[10] = 1.0f;
        matrix[11] = 0.0f;

        matrix[12] = 0.0f;
        matrix[13] = 0.0f;
        matrix[14] = 0.0f;
        matrix[15] = 1.0f;

        return;
    }


    void BuildPerspectiveFovLHMatrix(Matrix4X4f& matrix, const float fieldOfView, const float screenAspect, const float screenNear, const float screenDepth)
    {
        matrix[0] = 1.0f / (screenAspect * tanf(fieldOfView * 0.5f));
        matrix[1] = 0.0f;
        matrix[2] = 0.0f;
        matrix[3] = 0.0f;

        matrix[4] = 0.0f;
        matrix[5] = 1.0f / tanf(fieldOfView * 0.5f);
        matrix[6] = 0.0f;
        matrix[7] = 0.0f;

        matrix[8] = 0.0f;
        matrix[9] = 0.0f;
        matrix[10] = screenDepth / (screenDepth - screenNear);
        matrix[11] = 1.0f;

        matrix[12] = 0.0f;
        matrix[13] = 0.0f;
        matrix[14] = (-screenNear * screenDepth) / (screenDepth - screenNear);
        matrix[15] = 0.0f;

        return;
    }


    void MatrixRotationY(Matrix4X4f& matrix, const float angle)
    {
        matrix[0] = cosf(angle);
        matrix[1] = 0.0f;
        matrix[2] = -sinf(angle);
        matrix[3] = 0.0f;

        matrix[4] = 0.0f;
        matrix[5] = 1.0f;
        matrix[6] = 0.0f;
        matrix[7] = 0.0f;

        matrix[8] = sinf(angle);
        matrix[9] = 0.0f;
        matrix[10] = cosf(angle);
        matrix[11] = 0.0f;

        matrix[12] = 0.0f;
        matrix[13] = 0.0f;
        matrix[14] = 0.0f;
        matrix[15] = 1.0f;

        return;
    }


    void MatrixTranslation(Matrix4X4f& matrix, const float x, const float y, const float z)
    {
        matrix[0] = 1.0f;
        matrix[1] = 0.0f;
        matrix[2] = 0.0f;
        matrix[3] = 0.0f;

        matrix[4] = 0.0f;
        matrix[5] = 1.0f;
        matrix[6] = 0.0f;
        matrix[7] = 0.0f;

        matrix[8] = 0.0f;
        matrix[9] = 0.0f;
        matrix[10] = 1.0f;
        matrix[11] = 0.0f;

        matrix[12] = x;
        matrix[13] = y;
        matrix[14] = z;
        matrix[15] = 1.0f;

        return;
    }


    void MatrixRotationZ(Matrix4X4f& matrix, const float angle)
    {
        matrix[0] = cosf(angle);
        matrix[1] = -sinf(angle);
        matrix[2] = 0.0f;
        matrix[3] = 0.0f;

        matrix[4] = sinf(angle);
        matrix[5] = cosf(angle);
        matrix[6] = 0.0f;
        matrix[7] = 0.0f;

        matrix[8] = 0.0f;
        matrix[9] = 0.0f;
        matrix[10] = 1.0f;
        matrix[11] = 0.0f;

        matrix[12] = 0.0f;
        matrix[13] = 0.0f;
        matrix[14] = 0.0f;
        matrix[15] = 1.0f;

        return;
    }


    void MatrixMultiply(Matrix4X4f& result, const Matrix4X4f& matrix1, const Matrix4X4f& matrix2)
    {
        result[0] = (matrix1[0] * matrix2[0]) + (matrix1[1] * matrix2[4]) + (matrix1[2] * matrix2[8]) + (matrix1[3] * matrix2[12]);
        result[1] = (matrix1[0] * matrix2[1]) + (matrix1[1] * matrix2[5]) + (matrix1[2] * matrix2[9]) + (matrix1[3] * matrix2[13]);
        result[2] = (matrix1[0] * matrix2[2]) + (matrix1[1] * matrix2[6]) + (matrix1[2] * matrix2[10]) + (matrix1[3] * matrix2[14]);
        result[3] = (matrix1[0] * matrix2[3]) + (matrix1[1] * matrix2[7]) + (matrix1[2] * matrix2[11]) + (matrix1[3] * matrix2[15]);

        result[4] = (matrix1[4] * matrix2[0]) + (matrix1[5] * matrix2[4]) + (matrix1[6] * matrix2[8]) + (matrix1[7] * matrix2[12]);
        result[5] = (matrix1[4] * matrix2[1]) + (matrix1[5] * matrix2[5]) + (matrix1[6] * matrix2[9]) + (matrix1[7] * matrix2[13]);
        result[6] = (matrix1[4] * matrix2[2]) + (matrix1[5] * matrix2[6]) + (matrix1[6] * matrix2[10]) + (matrix1[7] * matrix2[14]);
        result[7] = (matrix1[4] * matrix2[3]) + (matrix1[5] * matrix2[7]) + (matrix1[6] * matrix2[11]) + (matrix1[7] * matrix2[15]);

        result[8] = (matrix1[8] * matrix2[0]) + (matrix1[9] * matrix2[4]) + (matrix1[10] * matrix2[8]) + (matrix1[11] * matrix2[12]);
        result[9] = (matrix1[8] * matrix2[1]) + (matrix1[9] * matrix2[5]) + (matrix1[10] * matrix2[9]) + (matrix1[11] * matrix2[13]);
        result[10] = (matrix1[8] * matrix2[2]) + (matrix1[9] * matrix2[6]) + (matrix1[10] * matrix2[10]) + (matrix1[11] * matrix2[14]);
        result[11] = (matrix1[8] * matrix2[3]) + (matrix1[9] * matrix2[7]) + (matrix1[10] * matrix2[11]) + (matrix1[11] * matrix2[15]);

        result[12] = (matrix1[12] * matrix2[0]) + (matrix1[13] * matrix2[4]) + (matrix1[14] * matrix2[8]) + (matrix1[15] * matrix2[12]);
        result[13] = (matrix1[12] * matrix2[1]) + (matrix1[13] * matrix2[5]) + (matrix1[14] * matrix2[9]) + (matrix1[15] * matrix2[13]);
        result[14] = (matrix1[12] * matrix2[2]) + (matrix1[13] * matrix2[6]) + (matrix1[14] * matrix2[10]) + (matrix1[15] * matrix2[14]);
        result[15] = (matrix1[12] * matrix2[3]) + (matrix1[13] * matrix2[7]) + (matrix1[14] * matrix2[11]) + (matrix1[15] * matrix2[15]);

        return;
    }

    inline void CrossProduct(Matrix3X3f& result, const Matrix3X3f matrix1, const Matrix3X3f matrix2)
    {
        ispc::CrossProduct(matrix1.data, matrix2.data, result.data);
    }

    inline void DotProduct(Matrix3X3f& result, const Matrix3X3f matrix1, const Matrix3X3f matrix2)
    {
        ispc::DotProduct(matrix1.data, matrix2.data, result.data, 3*3);
    }

    inline void DotProduct(Matrix4X4f& result, const Matrix4X4f matrix1, const Matrix4X4f matrix2)
    {
        ispc::DotProduct(matrix1.data, matrix2.data, result.data, 4*4);
    }

    inline void MulByElement(Matrix3X3f& result, const Matrix3X3f matrix1, const Matrix3X3f matrix2)
    {
        ispc::MulByElement(matrix1.data, matrix2.data, result.data, 3*3);
    }

    inline void MulByElement(Matrix4X4f& result, const Matrix4X4f matrix1, const Matrix4X4f matrix2)
    {
        ispc::MulByElement(matrix1.data, matrix2.data, result.data, 4*4);
    }

    inline void Transpose(Matrix3X3f& result, const Matrix3X3f matrix1)
    {
        ispc::Transpose(matrix1.data, result.data, 3, 3);
    }

    inline void Transpose(Matrix4X4f& result, const Matrix4X4f matrix1)
    {
        ispc::Transpose(matrix1.data, result.data, 4, 4);
    }

    inline void Normalize(Vector3f& result)
    {
        ispc::Normalize(result.data, 3);
    }

    inline void Normalize(Vector4f& result)
    {
        ispc::Normalize(result.data, 4);
    }
}

