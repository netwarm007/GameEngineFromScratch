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

struct main0_out
{
    float4 normal [[user(locn0)]];
    float4 normal_world [[user(locn1)]];
    float4 v [[user(locn2)]];
    float4 v_world [[user(locn3)]];
    float2 uv [[user(locn4)]];
    float3 TBN_0 [[user(locn5)]];
    float3 TBN_1 [[user(locn6)]];
    float3 TBN_2 [[user(locn7)]];
    float3 v_tangent [[user(locn8)]];
    float3 camPos_tangent [[user(locn9)]];
    float4 gl_Position [[position]];
};

struct main0_in
{
    float3 inputPosition [[attribute(0)]];
    float2 inputUV [[attribute(1)]];
};

vertex main0_out main0(main0_in in [[stage_in]], constant PerFrameConstants& _34 [[buffer(0)]])
{
    main0_out out = {};
    float3x3 TBN = {};
    out.v_world = float4(in.inputPosition, 1.0);
    out.v = _34.viewMatrix * out.v_world;
    out.gl_Position = _34.projectionMatrix * out.v;
    out.normal_world = float4(0.0, 0.0, 1.0, 0.0);
    out.normal = normalize(_34.viewMatrix * out.normal_world);
    float3 tangent = float3(1.0, 0.0, 0.0);
    float3 bitangent = float3(0.0, 1.0, 0.0);
    tangent = normalize(tangent - (out.normal_world.xyz * dot(tangent, out.normal_world.xyz)));
    bitangent = cross(out.normal_world.xyz, tangent);
    TBN = float3x3(float3(tangent), float3(bitangent), float3(out.normal_world.xyz));
    float3x3 TBN_trans = transpose(TBN);
    out.v_tangent = TBN_trans * out.v_world.xyz;
    out.camPos_tangent = TBN_trans * _34.camPos.xyz;
    out.uv.x = in.inputUV.x;
    out.uv.y = 1.0 - in.inputUV.y;
    out.TBN_0 = TBN[0];
    out.TBN_1 = TBN[1];
    out.TBN_2 = TBN[2];
    return out;
}

