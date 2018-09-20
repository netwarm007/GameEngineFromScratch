#version 310 es
#extension GL_EXT_geometry_shader : require
layout(triangles) in;
layout(max_vertices = 18, triangle_strip) out;

struct Light
{
    int lightType;
    float lightIntensity;
    int lightCastShadow;
    int lightShadowMapIndex;
    int lightAngleAttenCurveType;
    int lightDistAttenCurveType;
    vec2 lightSize;
    ivec4 lightGUID;
    vec4 lightPosition;
    vec4 lightColor;
    vec4 lightDirection;
    vec4 lightDistAttenCurveParams[2];
    vec4 lightAngleAttenCurveParams[2];
    mat4 lightVP;
    vec4 padding[2];
};

layout(binding = 2, std140) uniform ShadowMatrices
{
    mat4 shadowMatrices[6];
} _64;

struct gs_constant_t
{
    float layer_index;
};

uniform gs_constant_t u_gsPushConstants;

layout(location = 0) out vec4 FragPos;

void main()
{
    for (int face = 0; face < 6; face++)
    {
        gl_Layer = (int(u_gsPushConstants.layer_index) * 6) + face;
        for (int i = 0; i < 3; i++)
        {
            FragPos = gl_in[i].gl_Position;
            gl_Position = _64.shadowMatrices[face] * FragPos;
            EmitVertex();
        }
        EndPrimitive();
    }
}

