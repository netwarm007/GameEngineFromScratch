#ifndef __STDCBUFFER_H__
#define __STDCBUFFER_H__

#include "shader_base.h"

namespace My {
    struct BasicVertex
    {
        Vector3Type m_Position;
        Vector3Type m_Normal;
        Vector4Type m_Tangent;
        Vector2Type m_TextureUV;
    };

    unistruct Constants
    {
        Matrix4X4        m_modelView;
        Matrix4X4        m_modelViewProjection;
        Vector4Type      m_lightPosition;
        Vector4Type      m_lightColor;
        Vector4Type      m_ambientColor;
        Vector4Type      m_lightAttenuation;
    };
}

#endif // !__STDCBUFFER_H__
