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

struct _20
{
    float4x4 _m0;
    float4x4 _m1;
    float4 _m2;
    int _m3;
    Light _m4[100];
};

struct _Global
{
    float _m0;
};

fragment void terrain_frag_main()
{
    float4 v_world;
    float4 normal_world;
    float4 outputColor;
    float2 uv;
    float3x3 TBN;
    float3 v_tangent;
    float3 camPos_tangent;
}

