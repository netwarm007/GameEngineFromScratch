#ifndef __STDCBUFFER_H__
#define __STDCBUFFER_H__

#include "shader_base.h"

namespace My {
    struct BasicVertex
    {
        Vector3f m_Position;
        Vector3f m_Normal;
        Vector4f m_Tangent;
        Vector2f m_TextureUV;
    };

    unistruct Constants
    {
        Matrix4X4f       m_modelView;
        Matrix4X4f       m_modelViewProjection;
        Vector4f         m_lightPosition;
        Vector4f         m_lightColor;
        Vector4f         m_ambientColor;
        Vector4f         m_lightAttenuation;
    };
}

#endif // !__STDCBUFFER_H__
