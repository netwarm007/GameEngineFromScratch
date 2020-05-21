#include "SceneObjectMesh.hpp"

using namespace My;
using namespace std;

BoundingBox SceneObjectMesh::GetBoundingBox() const {
    Vector3f bbmin(numeric_limits<float>::max());
    Vector3f bbmax(numeric_limits<float>::lowest());
    auto count = m_VertexArray.size();
    for (decltype(count) n = 0; n < count; n++) {
        if (m_VertexArray[n].GetAttributeName() == "position") {
            auto data_type = m_VertexArray[n].GetDataType();
            auto vertices_count = m_VertexArray[n].GetVertexCount();
            auto data = m_VertexArray[n].GetData();
            for (decltype(vertices_count) i = 0; i < vertices_count; i++) {
                switch (data_type) {
                    case VertexDataType::kVertexDataTypeFloat3: {
                        const Vector3f* vertex =
                            reinterpret_cast<const Vector3f*>(data) + i;
                        bbmin[0] = (bbmin[0] < vertex->data[0])
                                       ? bbmin[0]
                                       : vertex->data[0];
                        bbmin[1] = (bbmin[1] < vertex->data[1])
                                       ? bbmin[1]
                                       : vertex->data[1];
                        bbmin[2] = (bbmin[2] < vertex->data[2])
                                       ? bbmin[2]
                                       : vertex->data[2];
                        bbmax[0] = (bbmax[0] > vertex->data[0])
                                       ? bbmax[0]
                                       : vertex->data[0];
                        bbmax[1] = (bbmax[1] > vertex->data[1])
                                       ? bbmax[1]
                                       : vertex->data[1];
                        bbmax[2] = (bbmax[2] > vertex->data[2])
                                       ? bbmax[2]
                                       : vertex->data[2];
                        break;
                    }
                    case VertexDataType::kVertexDataTypeDouble3: {
                        const Vector3* vertex =
                            reinterpret_cast<const Vector3*>(data) + i;
                        bbmin[0] = static_cast<float>(
                            (bbmin[0] < vertex->data[0]) ? bbmin[0]
                                                         : vertex->data[0]);
                        bbmin[1] = static_cast<float>(
                            (bbmin[1] < vertex->data[1]) ? bbmin[1]
                                                         : vertex->data[1]);
                        bbmin[2] = static_cast<float>(
                            (bbmin[2] < vertex->data[2]) ? bbmin[2]
                                                         : vertex->data[2]);
                        bbmax[0] = static_cast<float>(
                            (bbmax[0] > vertex->data[0]) ? bbmax[0]
                                                         : vertex->data[0]);
                        bbmax[1] = static_cast<float>(
                            (bbmax[1] > vertex->data[1]) ? bbmax[1]
                                                         : vertex->data[1]);
                        bbmax[2] = static_cast<float>(
                            (bbmax[2] > vertex->data[2]) ? bbmax[2]
                                                         : vertex->data[2]);
                        break;
                    }
                    default:
                        assert(0);
                }
            }
        }
    }

    BoundingBox result;
    result.extent = (bbmax - bbmin) * 0.5f;
    result.centroid = (bbmax + bbmin) * 0.5f;

    return result;
}

ConvexHull SceneObjectMesh::GetConvexHull() const {
    ConvexHull hull;

    auto count = m_VertexArray.size();
    for (decltype(count) n = 0; n < count; n++) {
        if (m_VertexArray[n].GetAttributeName() == "position") {
            auto data_type = m_VertexArray[n].GetDataType();
            auto vertices_count = m_VertexArray[n].GetVertexCount();
            auto data = m_VertexArray[n].GetData();
            for (decltype(vertices_count) i = 0; i < vertices_count; i++) {
                switch (data_type) {
                    case VertexDataType::kVertexDataTypeFloat3: {
                        const Vector3f* vertex =
                            reinterpret_cast<const Vector3f*>(data) + i;
                        hull.AddPoint(*vertex);
                        break;
                    }
                    case VertexDataType::kVertexDataTypeDouble3: {
                        const Vector3* vertex =
                            reinterpret_cast<const Vector3*>(data) + i;
                        hull.AddPoint(*vertex);
                        break;
                    }
                    default:
                        assert(0);
                }
            }
        }
    }

    // calculate the convex hull
    hull.Iterate();

    return hull;
}
