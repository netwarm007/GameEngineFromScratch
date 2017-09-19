#pragma once
#include "portable.h"

namespace My {
    typedef ENUM(IndexSize) {
            kIndexSize16 = 0x00000010,
            kIndexSize32 = 0x00000020,
            kIndexSizePlatformDependent = 0x10000000, // start point of platform specific value
    } IndexSize;

    typedef ENUM(PrimitiveType) {
            kPrimitiveTypeNone                  = 0x00000000, ///< No particular primitive type.
            kPrimitiveTypePointList             = 0x00000001, ///< For N>=0, vertex N renders a point.
            kPrimitiveTypeLineList              = 0x00000002, ///< For N>=0, vertices [N*2+0, N*2+1] render a line.
            kPrimitiveTypeLineStrip             = 0x00000003, ///< For N>=0, vertices [N, N+1] render a line.
            kPrimitiveTypeTriList               = 0x00000004, ///< For N>=0, vertices [N*3+0, N*3+1, N*3+2] render a triangle.
            kPrimitiveTypeTriFan                = 0x00000005, ///< For N>=0, vertices [0, (N+1)%M, (N+2)%M] render a triangle, where M is the vertex count.
            kPrimitiveTypeTriStrip              = 0x00000006, ///< For N>=0, vertices [N*2+0, N*2+1, N*2+2] and [N*2+2, N*2+1, N*2+3] render triangles.
            kPrimitiveTypePatch                 = 0x00000009, ///< Used for tessellation.
            kPrimitiveTypeLineListAdjacency     = 0x0000000a, ///< For N>=0, vertices [N*4..N*4+3] render a line from [1, 2]. Lines [0, 1] and [2, 3] are adjacent to the rendered line.
            kPrimitiveTypeLineStripAdjacency    = 0x0000000b, ///< For N>=0, vertices [N+1, N+2] render a line. Lines [N, N+1] and [N+2, N+3] are adjacent to the rendered line.
            kPrimitiveTypeTriListAdjacency      = 0x0000000c, ///< For N>=0, vertices [N*6..N*6+5] render a triangle from [0, 2, 4]. Triangles [0, 1, 2] [4, 2, 3] and [5, 0, 4] are adjacent to the rendered triangle.
            kPrimitiveTypeTriStripAdjacency     = 0x0000000d, ///< For N>=0, vertices [N*4..N*4+6] render a triangle from [0, 2, 4] and [4, 2, 6]. Odd vertices Nodd form adjacent triangles with indices min(Nodd+1,Nlast) and max(Nodd-3,Nfirst).
            kPrimitiveTypeRectList              = 0x00000011, ///< For N>=0, vertices [N*3+0, N*3+1, N*3+2] render a screen-aligned rectangle. 0 is upper-left, 1 is upper-right, and 2 is the lower-left corner.
            kPrimitiveTypeLineLoop              = 0x00000012, ///< Like <c>kPrimitiveTypeLineStrip</c>, but the first and last vertices also render a line.
            kPrimitiveTypeQuadList              = 0x00000013, ///< For N>=0, vertices [N*4+0, N*4+1, N*4+2] and [N*4+0, N*4+2, N*4+3] render triangles.
            kPrimitiveTypeQuadStrip             = 0x00000014, ///< For N>=0, vertices [N*2+0, N*2+1, N*2+3] and [N*2+0, N*2+3, N*2+2] render triangles.
            kPrimitiveTypePolygon               = 0x00000015, ///< For N>=0, vertices [0, N+1, N+2] render a triangle.
    } PrimitiveType;

    class Mesh
    {
    public:
        Mesh() 
	: m_vertexBuffer(0)
	, m_vertexBufferSize(0)
	, m_vertexCount(0)
	, m_vertexAttributeCount(0)
	, m_indexBuffer(0)
	, m_indexBufferSize(0)
	, m_indexCount(0)
	, m_indexType(IndexSize::kIndexSize16)
    {};

    public:
        enum { kMaxiumVertexBufferElements = 16 };

        void* m_vertexBuffer;
        uint32_t m_vertexBufferSize;
        uint32_t m_vertexCount;
        uint32_t m_vertexAttributeCount;
        uint8_t  m_reserved0[4];
        void* m_indexBuffer;
        uint32_t m_indexBufferSize;
        uint32_t m_indexCount;
        IndexSize m_indexType;
    };

    class SimpleMesh : public Mesh
    {
    public:
        SimpleMesh() : Mesh()
	        , m_primitiveType(PrimitiveType::kPrimitiveTypeTriList) {};
            
        uint32_t m_vertexStride; // in bytes
        PrimitiveType m_primitiveType;
        uint32_t m_reserved[2];
    };
}

