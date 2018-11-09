#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct gs_constant_t
{
    float layer_index;
};

struct ShadowMatrices
{
    float4x4 shadowMatrices[6];
};

struct Light
{
    float lightIntensity;
    int lightType;
    int lightCastShadow;
    int lightShadowMapIndex;
    int lightAngleAttenCurveType;
    int lightDistAttenCurveType;
    float2 lightSize;
    int4 lightGUID;
    float4 lightPosition;
    float4 lightColor;
    float4 lightDirection;
    float4 lightDistAttenCurveParams[2];
    float4 lightAngleAttenCurveParams[2];
    float4x4 lightVP;
    float4 padding[2];
};

struct PerFrameConstants
{
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
    float4 camPos;
    int numLights;
    Light allLights[100];
};

struct PerBatchConstants
{
    float4x4 modelMatrix;
};

struct shadowmap_omni_gs_main_out
{
    float4 FragPos;
    int gl_Layer;
    float4 gl_Position;
};

unknown shadowmap_omni_gs_main_out shadowmap_omni_gs_main(constant gs_constant_t& u_gsPushConstants [[buffer(0)]], constant ShadowMatrices& _64 [[buffer(2)]], float4 gl_in [[position]])
{
    shadowmap_omni_gs_main_out out = {};
    for (int face = 0; face < 6; face++)
    {
        out.gl_Layer = (int(u_gsPushConstants.layer_index) * 6) + face;
        for (int i = 0; i < 3; i++)
        {
            out.FragPos = gl_in[i].out.gl_Position;
            out.gl_Position = _64.shadowMatrices[face] * out.FragPos;
            EmitVertex();
        }
        EndPrimitive();
    }
    return out;
}

