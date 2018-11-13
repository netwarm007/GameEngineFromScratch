#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct PerBatchConstants
{
    float4x4 modelMatrix;
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

struct main0_out
{
    float4 gl_Position [[position]];
};

struct main0_in
{
    float3 inputPosition [[attribute(0)]];
};

vertex main0_out main0(main0_in in [[stage_in]], constant PerBatchConstants& _50 [[buffer(1)]], texture2d<float> terrainHeightMap [[texture(11)]], sampler terrainHeightMapSmplr [[sampler(11)]])
{
    main0_out out = {};
    float height = terrainHeightMap.sample(terrainHeightMapSmplr, (in.inputPosition.xy / float2(10800.0)), level(0.0)).x * 10.0;
    float4 displaced = float4(in.inputPosition.xy, height, 1.0);
    out.gl_Position = _50.modelMatrix * displaced;
    return out;
}

