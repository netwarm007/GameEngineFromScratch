#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

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

struct skybox_vert_main_out
{
    float3 UVW [[user(locn0)]];
    float4 gl_Position [[position]];
};

struct skybox_vert_main_in
{
    float3 inputPosition [[attribute(0)]];
};

vertex skybox_vert_main_out skybox_vert_main(skybox_vert_main_in in [[stage_in]], constant PerFrameConstants& _30 [[buffer(0)]])
{
    skybox_vert_main_out out = {};
    out.UVW = in.inputPosition;
    float4x4 matrix = _30.viewMatrix;
    matrix[3].x = 0.0;
    matrix[3].y = 0.0;
    matrix[3].z = 0.0;
    float4 pos = (_30.projectionMatrix * matrix) * float4(in.inputPosition, 1.0);
    out.gl_Position = pos.xyww;
    return out;
}

