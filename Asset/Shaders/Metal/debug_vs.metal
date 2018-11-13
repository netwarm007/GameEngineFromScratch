#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct Light
{
    int lightType;
    float lightIntensity;
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

struct main0_out
{
    float4 gl_Position [[position]];
    float gl_PointSize [[point_size]];
};

struct main0_in
{
    float3 inputPosition [[attribute(0)]];
};

vertex main0_out main0(main0_in in [[stage_in]], constant PerFrameConstants& _42 [[buffer(0)]])
{
    main0_out out = {};
    out.gl_PointSize = 5.0;
    float4 v = float4(in.inputPosition, 1.0);
    v = _42.viewMatrix * v;
    out.gl_Position = _42.projectionMatrix * v;
    return out;
}

