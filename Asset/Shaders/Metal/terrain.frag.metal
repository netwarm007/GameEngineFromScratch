#pragma clang diagnostic ignored "-Wmissing-prototypes"

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

struct Light_1
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

struct PerBatchConstants
{
    float4x4 modelMatrix;
};

constant float _52 = {};

struct terrain_frag_main_out
{
    float4 outputColor [[color(0)]];
};

struct terrain_frag_main_in
{
    float4 normal_world [[user(locn1)]];
    float4 v_world [[user(locn3)]];
};

// Implementation of an array copy function to cover GLSL's ability to copy an array via assignment.
template<typename T, uint N>
void spvArrayCopy(thread T (&dst)[N], thread const T (&src)[N])
{
    for (uint i = 0; i < N; dst[i] = src[i], i++);
}

// An overload for constant arrays.
template<typename T, uint N>
void spvArrayCopyConstant(thread T (&dst)[N], constant T (&src)[N])
{
    for (uint i = 0; i < N; dst[i] = src[i], i++);
}

float linear_interpolate(thread const float& t, thread const float& begin, thread const float& end)
{
    if (t < begin)
    {
        return 1.0;
    }
    else
    {
        if (t > end)
        {
            return 0.0;
        }
        else
        {
            return (end - t) / (end - begin);
        }
    }
}

float apply_atten_curve(thread const float& dist, thread const int& atten_curve_type, thread const float4 (&atten_params)[2])
{
    float atten = 1.0;
    switch (atten_curve_type)
    {
        case 1:
        {
            float begin_atten = atten_params[0].x;
            float end_atten = atten_params[0].y;
            float param = dist;
            float param_1 = begin_atten;
            float param_2 = end_atten;
            atten = linear_interpolate(param, param_1, param_2);
            break;
        }
        case 2:
        {
            float begin_atten_1 = atten_params[0].x;
            float end_atten_1 = atten_params[0].y;
            float param_3 = dist;
            float param_4 = begin_atten_1;
            float param_5 = end_atten_1;
            float tmp = linear_interpolate(param_3, param_4, param_5);
            atten = (3.0 * pow(tmp, 2.0)) - (2.0 * pow(tmp, 3.0));
            break;
        }
        case 3:
        {
            float scale = atten_params[0].x;
            float offset = atten_params[0].y;
            float kl = atten_params[0].z;
            float kc = atten_params[0].w;
            atten = clamp((scale / ((kl * dist) + (kc * scale))) + offset, 0.0, 1.0);
            break;
        }
        case 4:
        {
            float scale_1 = atten_params[0].x;
            float offset_1 = atten_params[0].y;
            float kq = atten_params[0].z;
            float kl_1 = atten_params[0].w;
            float kc_1 = atten_params[1].x;
            atten = clamp(pow(scale_1, 2.0) / ((((kq * pow(dist, 2.0)) + ((kl_1 * dist) * scale_1)) + (kc_1 * pow(scale_1, 2.0))) + offset_1), 0.0, 1.0);
            break;
        }
        case 0:
        {
            break;
        }
        default:
        {
            break;
        }
    }
    return atten;
}

fragment terrain_frag_main_out terrain_frag_main(terrain_frag_main_in in [[stage_in]], constant PerFrameConstants& _193 [[buffer(0)]])
{
    terrain_frag_main_out out = {};
    float3 Lo = float3(0.0);
    for (int i = 0; i < _193.numLights; i++)
    {
        Light_1 light;
        light.lightIntensity = _193.allLights[i].lightIntensity;
        light.lightType = _193.allLights[i].lightType;
        light.lightCastShadow = _193.allLights[i].lightCastShadow;
        light.lightShadowMapIndex = _193.allLights[i].lightShadowMapIndex;
        light.lightAngleAttenCurveType = _193.allLights[i].lightAngleAttenCurveType;
        light.lightDistAttenCurveType = _193.allLights[i].lightDistAttenCurveType;
        light.lightSize = _193.allLights[i].lightSize;
        light.lightGUID = _193.allLights[i].lightGUID;
        light.lightPosition = _193.allLights[i].lightPosition;
        light.lightColor = _193.allLights[i].lightColor;
        light.lightDirection = _193.allLights[i].lightDirection;
        light.lightDistAttenCurveParams[0] = _193.allLights[i].lightDistAttenCurveParams[0];
        light.lightDistAttenCurveParams[1] = _193.allLights[i].lightDistAttenCurveParams[1];
        light.lightAngleAttenCurveParams[0] = _193.allLights[i].lightAngleAttenCurveParams[0];
        light.lightAngleAttenCurveParams[1] = _193.allLights[i].lightAngleAttenCurveParams[1];
        light.lightVP = _193.allLights[i].lightVP;
        light.padding[0] = _193.allLights[i].padding[0];
        light.padding[1] = _193.allLights[i].padding[1];
        float3 L = normalize(light.lightPosition.xyz - in.v_world.xyz);
        float3 N = in.normal_world.xyz;
        float NdotL = max(dot(N, L), 0.0);
        float visibility = 1.0;
        float lightToSurfDist = length(L);
        float lightToSurfAngle = acos(dot(-L, light.lightDirection.xyz));
        float param = lightToSurfAngle;
        int param_1 = light.lightAngleAttenCurveType;
        float4 param_2[2];
        spvArrayCopy(param_2, light.lightAngleAttenCurveParams);
        float atten = apply_atten_curve(param, param_1, param_2);
        float param_3 = lightToSurfDist;
        int param_4 = light.lightDistAttenCurveType;
        float4 param_5[2];
        spvArrayCopy(param_5, light.lightDistAttenCurveParams);
        atten *= apply_atten_curve(param_3, param_4, param_5);
        float3 radiance = light.lightColor.xyz * (light.lightIntensity * atten);
        Lo += ((radiance * NdotL) * visibility);
    }
    out.outputColor = float4(Lo, 1.0);
    return out;
}

