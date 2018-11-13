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

struct terrain_vert_main_out
{
    float4 gl_Position [[position]];
};

struct terrain_vert_main_in
{
    float3 inputPosition [[attribute(0)]];
};

vertex terrain_vert_main_out terrain_vert_main(terrain_vert_main_in in [[stage_in]], texture2d<float> terrainHeightMap [[texture(11)]], sampler terrainHeightMapSmplr [[sampler(11)]])
{
    terrain_vert_main_out out = {};
    float height = terrainHeightMap.sample(terrainHeightMapSmplr, in.inputPosition.xy, level(0.0)).x;
    float4 displaced = float4(in.inputPosition.xy, height, 1.0);
    out.gl_Position = displaced;
    return out;
}

