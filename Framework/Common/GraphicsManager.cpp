#include <iostream>
#include "GraphicsManager.hpp"
#include "cbuffer.h"

using namespace My;
using namespace std;

int GraphicsManager::Initialize()
{
    int result = 0;
    return result;
}

void GraphicsManager::Finalize()
{
}

void GraphicsManager::Tick()
{
}

void GraphicsManager::DrawSingleMesh(const Mesh& mesh)
{
    cout << "DrawSingleMesh"                                        << endl;
    cout << "--------------"                                        << endl;
    cout << "Vertex Buffer: "           << mesh.m_vertexBuffer      << endl;
    cout << "Vertex Buffer Size: "      << mesh.m_vertexBufferSize  << endl;
    cout << "Vertex Count: "            << mesh.m_vertexCount       << endl;
    cout << "Vertex Attribute Count: "  << mesh.m_vertexAttributeCount << endl;
    cout << endl;
    cout << "Index Buffer: "            << mesh.m_indexBuffer       << endl;
    cout << "Index Buffer Size: "       << mesh.m_indexBufferSize   << endl;
    cout << "Index Count: "             << mesh.m_indexCount << endl;
    // cout << "Index Type: "              << mesh.m_indexType         << endl;
}

