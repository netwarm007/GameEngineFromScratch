#pragma once
#include "geommath.hpp"
#include "portable.hpp"

namespace My {
    ENUM(IndexDataType) {
        kIndexDataTypeInt8  = "I8  "_i32,
        kIndexDataTypeInt16 = "I16 "_i32,
        kIndexDataTypeInt32 = "I32 "_i32,
        kIndexDataTypeInt64 = "I64 "_i32,
    };

    std::ostream& operator<<(std::ostream& out, IndexDataType type);

    ENUM(VertexDataType) {
        kVertexDataTypeFloat1    = "FLT1"_i32,
        kVertexDataTypeFloat2    = "FLT2"_i32,
        kVertexDataTypeFloat3    = "FLT3"_i32,
        kVertexDataTypeFloat4    = "FLT4"_i32,
        kVertexDataTypeDouble1   = "DUB1"_i32,
        kVertexDataTypeDouble2   = "DUB2"_i32,
        kVertexDataTypeDouble3   = "DUB3"_i32,
        kVertexDataTypeDouble4   = "DUB4"_i32
    };

    std::ostream& operator<<(std::ostream& out, VertexDataType type);

	ENUM(PrimitiveType) {
		kPrimitiveTypeNone = "NONE"_i32,        ///< No particular primitive type.
		kPrimitiveTypePointList = "PLST"_i32,   ///< For N>=0, vertex N renders a point.
		kPrimitiveTypeLineList = "LLST"_i32,    ///< For N>=0, vertices [N*2+0, N*2+1] render a line.
		kPrimitiveTypeLineStrip = "LSTR"_i32,   ///< For N>=0, vertices [N, N+1] render a line.
		kPrimitiveTypeTriList = "TLST"_i32,     ///< For N>=0, vertices [N*3+0, N*3+1, N*3+2] render a triangle.
		kPrimitiveTypeTriFan = "TFAN"_i32,      ///< For N>=0, vertices [0, (N+1)%M, (N+2)%M] render a triangle, where M is the vertex count.
		kPrimitiveTypeTriStrip = "TSTR"_i32,    ///< For N>=0, vertices [N*2+0, N*2+1, N*2+2] and [N*2+2, N*2+1, N*2+3] render triangles.
		kPrimitiveTypePatch = "PACH"_i32,       ///< Used for tessellation.
		kPrimitiveTypeLineListAdjacency = "LLSA"_i32,       ///< For N>=0, vertices [N*4..N*4+3] render a line from [1, 2]. Lines [0, 1] and [2, 3] are adjacent to the rendered line.
		kPrimitiveTypeLineStripAdjacency = "LSTA"_i32,      ///< For N>=0, vertices [N+1, N+2] render a line. Lines [N, N+1] and [N+2, N+3] are adjacent to the rendered line.
		kPrimitiveTypeTriListAdjacency = "TLSA"_i32,        ///< For N>=0, vertices [N*6..N*6+5] render a triangle from [0, 2, 4]. Triangles [0, 1, 2] [4, 2, 3] and [5, 0, 4] are adjacent to the rendered triangle.
		kPrimitiveTypeTriStripAdjacency = "TSTA"_i32,       ///< For N>=0, vertices [N*4..N*4+6] render a triangle from [0, 2, 4] and [4, 2, 6]. Odd vertices Nodd form adjacent triangles with indices min(Nodd+1,Nlast) and max(Nodd-3,Nfirst).
		kPrimitiveTypeRectList = "RLST"_i32,    ///< For N>=0, vertices [N*3+0, N*3+1, N*3+2] render a screen-aligned rectangle. 0 is upper-left, 1 is upper-right, and 2 is the lower-left corner.
		kPrimitiveTypeLineLoop = "LLOP"_i32,    ///< Like <c>kPrimitiveTypeLineStrip</c>, but the first and last vertices also render a line.
		kPrimitiveTypeQuadList = "QLST"_i32,    ///< For N>=0, vertices [N*4+0, N*4+1, N*4+2] and [N*4+0, N*4+2, N*4+3] render triangles.
		kPrimitiveTypeQuadStrip = "QSTR"_i32,   ///< For N>=0, vertices [N*2+0, N*2+1, N*2+3] and [N*2+0, N*2+3, N*2+2] render triangles.
		kPrimitiveTypePolygon = "POLY"_i32,     ///< For N>=0, vertices [0, N+1, N+2] render a triangle.
	};

    struct BoundingBox {
        Vector3f centroid;
        Vector3f extent;
    };

    std::ostream& operator<<(std::ostream& out, PrimitiveType type);
  
    class SceneObjectTexture;

    template <typename T>
    struct ParameterValueMap
    {
        T Value;
        std::shared_ptr<SceneObjectTexture> ValueMap;

        ParameterValueMap() = default;

        ParameterValueMap(const T value) : Value(value), ValueMap(nullptr) {};
        ParameterValueMap(const std::shared_ptr<SceneObjectTexture>& value) : ValueMap(value) {};

        ParameterValueMap(const ParameterValueMap<T>& rhs) = default;

        ParameterValueMap(ParameterValueMap<T>&& rhs) = default;

        ParameterValueMap& operator=(const ParameterValueMap<T>& rhs) = default;
        ParameterValueMap& operator=(ParameterValueMap<T>&& rhs) = default;
        ParameterValueMap& operator=(const std::shared_ptr<SceneObjectTexture>& rhs) 
        {
            ValueMap = rhs;
            return *this;
        };

        ~ParameterValueMap() = default;

        friend std::ostream& operator<<(std::ostream& out, const ParameterValueMap<T>& obj)
        {
            out << "Parameter Value: " << obj.Value << std::endl;
            if (obj.ValueMap) {
                out << "Parameter Map: " << *obj.ValueMap << std::endl;
            }

            return out;
        }
    };

    typedef ParameterValueMap<Vector4f> Color;
    typedef ParameterValueMap<Vector3f> Normal;
    typedef ParameterValueMap<float>    Parameter;

}