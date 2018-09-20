#pragma clang diagnostic ignored "-Wmissing-prototypes"

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

struct Light_1
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
    Light_1 allLights[100];
};

struct constants_t
{
    float4 ambientColor;
    float4 specularColor;
    float specularPower;
};

struct Light_2
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

struct PerBatchConstants
{
    float4x4 modelMatrix;
};

constant float2 _354[4] = {float2(-0.94201624393463134765625, -0.39906215667724609375), float2(0.94558608531951904296875, -0.768907248973846435546875), float2(-0.094184100627899169921875, -0.929388701915740966796875), float2(0.34495937824249267578125, 0.29387760162353515625)};

constant float _131 = {};

struct main0_out
{
    float4 outputColor [[color(0)]];
};

struct main0_in
{
    float4 normal [[user(locn0)]];
    float4 normal_world [[user(locn1)]];
    float4 v [[user(locn2)]];
    float4 v_world [[user(locn3)]];
    float2 uv [[user(locn4)]];
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

float3 projectOnPlane(thread const float3& point, thread const float3& center_of_plane, thread const float3& normal_of_plane)
{
    return point - (normal_of_plane * dot(point - center_of_plane, normal_of_plane));
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

bool isAbovePlane(thread const float3& point, thread const float3& center_of_plane, thread const float3& normal_of_plane)
{
    return dot(point - center_of_plane, normal_of_plane) > 0.0;
}

float3 linePlaneIntersect(thread const float3& line_start, thread const float3& line_dir, thread const float3& center_of_plane, thread const float3& normal_of_plane)
{
    return line_start + (line_dir * (dot(center_of_plane - line_start, normal_of_plane) / dot(line_dir, normal_of_plane)));
}

float3 apply_areaLight(thread const Light& light, thread float4& normal, constant PerFrameConstants& v_500, thread float4& v, thread texture2d<float> diffuseMap, thread const sampler diffuseMapSmplr, thread float2& uv, constant constants_t& u_pushConstants)
{
    float3 N = normalize(normal.xyz);
    float3 right = normalize((v_500.viewMatrix * float4(1.0, 0.0, 0.0, 0.0)).xyz);
    float3 pnormal = normalize((v_500.viewMatrix * light.lightDirection).xyz);
    float3 ppos = (v_500.viewMatrix * light.lightPosition).xyz;
    float3 up = normalize(cross(pnormal, right));
    right = normalize(cross(up, pnormal));
    float width = light.lightSize.x;
    float height = light.lightSize.y;
    float3 param = v.xyz;
    float3 param_1 = ppos;
    float3 param_2 = pnormal;
    float3 projection = projectOnPlane(param, param_1, param_2);
    float3 dir = projection - ppos;
    float2 diagonal = float2(dot(dir, right), dot(dir, up));
    float2 nearest2D = float2(clamp(diagonal.x, -width, width), clamp(diagonal.y, -height, height));
    float3 nearestPointInside = (ppos + (right * nearest2D.x)) + (up * nearest2D.y);
    float3 L = nearestPointInside - v.xyz;
    float lightToSurfDist = length(L);
    L = normalize(L);
    float param_3 = lightToSurfDist;
    int param_4 = light.lightDistAttenCurveType;
    float4 param_5[2];
    spvArrayCopy(param_5, light.lightDistAttenCurveParams);
    float atten = apply_atten_curve(param_3, param_4, param_5);
    float3 linearColor = float3(0.0);
    float pnDotL = dot(pnormal, -L);
    float nDotL = dot(N, L);
    float _749 = nDotL;
    bool _750 = _749 > 0.0;
    bool _761;
    if (_750)
    {
        float3 param_6 = v.xyz;
        float3 param_7 = ppos;
        float3 param_8 = pnormal;
        _761 = isAbovePlane(param_6, param_7, param_8);
    }
    else
    {
        _761 = _750;
    }
    if (_761)
    {
        float3 V = normalize(-v.xyz);
        float3 R = normalize((N * (2.0 * dot(V, N))) - V);
        float3 R2 = normalize((N * (2.0 * dot(L, N))) - L);
        float3 param_9 = v.xyz;
        float3 param_10 = R;
        float3 param_11 = ppos;
        float3 param_12 = pnormal;
        float3 E = linePlaneIntersect(param_9, param_10, param_11, param_12);
        float specAngle = clamp(dot(-R, pnormal), 0.0, 1.0);
        float3 dirSpec = E - ppos;
        float2 dirSpec2D = float2(dot(dirSpec, right), dot(dirSpec, up));
        float2 nearestSpec2D = float2(clamp(dirSpec2D.x, -width, width), clamp(dirSpec2D.y, -height, height));
        float specFactor = 1.0 - clamp(length(nearestSpec2D - dirSpec2D), 0.0, 1.0);
        float3 admit_light = light.lightColor.xyz * (light.lightIntensity * atten);
        linearColor = (diffuseMap.sample(diffuseMapSmplr, uv).xyz * nDotL) * pnDotL;
        linearColor += (((u_pushConstants.specularColor.xyz * pow(clamp(dot(R2, V), 0.0, 1.0), u_pushConstants.specularPower)) * specFactor) * specAngle);
        linearColor *= admit_light;
    }
    return linearColor;
}

float shadow_test(thread const float4& p, thread const Light& light, thread const float& cosTheta, thread texturecube_array<float> cubeShadowMap, thread const sampler cubeShadowMapSmplr, thread texture2d_array<float> shadowMap, thread const sampler shadowMapSmplr, thread texture2d_array<float> globalShadowMap, thread const sampler globalShadowMapSmplr)
{
    float4 v_light_space = light.lightVP * p;
    v_light_space /= float4(v_light_space.w);
    float visibility = 1.0;
    if (light.lightShadowMapIndex != (-1))
    {
        float bias0 = 0.0005000000237487256526947021484375 * tan(acos(cosTheta));
        bias0 = clamp(bias0, 0.0, 0.00999999977648258209228515625);
        float near_occ;
        switch (light.lightType)
        {
            case 0:
            {
                float3 L = p.xyz - light.lightPosition.xyz;
                near_occ = cubeShadowMap.sample(cubeShadowMapSmplr, float4(L, float(light.lightShadowMapIndex)).xyz, uint(round(float4(L, float(light.lightShadowMapIndex)).w))).x;
                if ((length(L) - (near_occ * 10.0)) > bias0)
                {
                    visibility -= 0.87999999523162841796875;
                }
                break;
            }
            case 1:
            {
                v_light_space = float4x4(float4(0.5, 0.0, 0.0, 0.0), float4(0.0, 0.5, 0.0, 0.0), float4(0.0, 0.0, 0.5, 0.0), float4(0.5, 0.5, 0.5, 1.0)) * v_light_space;
                for (int i = 0; i < 4; i++)
                {
                    float2 indexable[4] = {float2(-0.94201624393463134765625, -0.39906215667724609375), float2(0.94558608531951904296875, -0.768907248973846435546875), float2(-0.094184100627899169921875, -0.929388701915740966796875), float2(0.34495937824249267578125, 0.29387760162353515625)};
                    near_occ = shadowMap.sample(shadowMapSmplr, float3(v_light_space.xy + (indexable[i] / float2(700.0)), float(light.lightShadowMapIndex)).xy, uint(round(float3(v_light_space.xy + (indexable[i] / float2(700.0)), float(light.lightShadowMapIndex)).z))).x;
                    if ((v_light_space.z - near_occ) > bias0)
                    {
                        visibility -= 0.2199999988079071044921875;
                    }
                }
                break;
            }
            case 2:
            {
                v_light_space = float4x4(float4(0.5, 0.0, 0.0, 0.0), float4(0.0, 0.5, 0.0, 0.0), float4(0.0, 0.0, 0.5, 0.0), float4(0.5, 0.5, 0.5, 1.0)) * v_light_space;
                for (int i_1 = 0; i_1 < 4; i_1++)
                {
                    float2 indexable_1[4] = {float2(-0.94201624393463134765625, -0.39906215667724609375), float2(0.94558608531951904296875, -0.768907248973846435546875), float2(-0.094184100627899169921875, -0.929388701915740966796875), float2(0.34495937824249267578125, 0.29387760162353515625)};
                    near_occ = globalShadowMap.sample(globalShadowMapSmplr, float3(v_light_space.xy + (indexable_1[i_1] / float2(700.0)), float(light.lightShadowMapIndex)).xy, uint(round(float3(v_light_space.xy + (indexable_1[i_1] / float2(700.0)), float(light.lightShadowMapIndex)).z))).x;
                    if ((v_light_space.z - near_occ) > bias0)
                    {
                        visibility -= 0.2199999988079071044921875;
                    }
                }
                break;
            }
            case 3:
            {
                v_light_space = float4x4(float4(0.5, 0.0, 0.0, 0.0), float4(0.0, 0.5, 0.0, 0.0), float4(0.0, 0.0, 0.5, 0.0), float4(0.5, 0.5, 0.5, 1.0)) * v_light_space;
                for (int i_2 = 0; i_2 < 4; i_2++)
                {
                    float2 indexable_2[4] = {float2(-0.94201624393463134765625, -0.39906215667724609375), float2(0.94558608531951904296875, -0.768907248973846435546875), float2(-0.094184100627899169921875, -0.929388701915740966796875), float2(0.34495937824249267578125, 0.29387760162353515625)};
                    near_occ = shadowMap.sample(shadowMapSmplr, float3(v_light_space.xy + (indexable_2[i_2] / float2(700.0)), float(light.lightShadowMapIndex)).xy, uint(round(float3(v_light_space.xy + (indexable_2[i_2] / float2(700.0)), float(light.lightShadowMapIndex)).z))).x;
                    if ((v_light_space.z - near_occ) > bias0)
                    {
                        visibility -= 0.2199999988079071044921875;
                    }
                }
                break;
            }
        }
    }
    return visibility;
}

float3 apply_light(thread const Light& light, thread texturecube_array<float> cubeShadowMap, thread const sampler cubeShadowMapSmplr, thread texture2d_array<float> shadowMap, thread const sampler shadowMapSmplr, thread texture2d_array<float> globalShadowMap, thread const sampler globalShadowMapSmplr, thread float4& normal, constant PerFrameConstants& v_500, thread float4& v, thread float4& v_world, thread texture2d<float> diffuseMap, thread const sampler diffuseMapSmplr, thread float2& uv, constant constants_t& u_pushConstants)
{
    float3 N = normalize(normal.xyz);
    float3 light_dir = normalize((v_500.viewMatrix * light.lightDirection).xyz);
    float3 L;
    if (light.lightPosition.w == 0.0)
    {
        L = -light_dir;
    }
    else
    {
        L = (v_500.viewMatrix * light.lightPosition).xyz - v.xyz;
    }
    float lightToSurfDist = length(L);
    L = normalize(L);
    float cosTheta = clamp(dot(N, L), 0.0, 1.0);
    float visibility = shadow_test(v_world, light, cosTheta, cubeShadowMap, cubeShadowMapSmplr, shadowMap, shadowMapSmplr, globalShadowMap, globalShadowMapSmplr);
    float lightToSurfAngle = acos(dot(L, -light_dir));
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
    float3 R = normalize((N * (2.0 * dot(L, N))) - L);
    float3 V = normalize(-v.xyz);
    float3 admit_light = light.lightColor.xyz * (light.lightIntensity * atten);
    float3 linearColor = diffuseMap.sample(diffuseMapSmplr, uv).xyz * cosTheta;
    if (visibility > 0.20000000298023223876953125)
    {
        linearColor += (u_pushConstants.specularColor.xyz * pow(clamp(dot(R, V), 0.0, 1.0), u_pushConstants.specularPower));
    }
    linearColor *= admit_light;
    return linearColor * visibility;
}

float3 exposure_tone_mapping(thread const float3& color)
{
    return float3(1.0) - exp((-color) * 1.0);
}

float3 gamma_correction(thread const float3& color)
{
    return pow(color, float3(0.4545454680919647216796875));
}

fragment main0_out main0(main0_in in [[stage_in]], constant PerFrameConstants& v_500 [[buffer(0)]], constant constants_t& u_pushConstants [[buffer(0)]], texture2d<float> diffuseMap [[texture(0)]], texture2d_array<float> shadowMap [[texture(1)]], texture2d_array<float> globalShadowMap [[texture(2)]], texturecube_array<float> cubeShadowMap [[texture(3)]], texturecube_array<float> skybox [[texture(4)]], sampler diffuseMapSmplr [[sampler(0)]], sampler shadowMapSmplr [[sampler(1)]], sampler globalShadowMapSmplr [[sampler(2)]], sampler cubeShadowMapSmplr [[sampler(3)]], sampler skyboxSmplr [[sampler(4)]])
{
    main0_out out = {};
    float3 linearColor = float3(0.0);
    for (int i = 0; i < v_500.numLights; i++)
    {
        if (v_500.allLights[i].lightType == 3)
        {
            Light arg;
            arg.lightType = v_500.allLights[i].lightType;
            arg.lightIntensity = v_500.allLights[i].lightIntensity;
            arg.lightCastShadow = v_500.allLights[i].lightCastShadow;
            arg.lightShadowMapIndex = v_500.allLights[i].lightShadowMapIndex;
            arg.lightAngleAttenCurveType = v_500.allLights[i].lightAngleAttenCurveType;
            arg.lightDistAttenCurveType = v_500.allLights[i].lightDistAttenCurveType;
            arg.lightSize = v_500.allLights[i].lightSize;
            arg.lightGUID = v_500.allLights[i].lightGUID;
            arg.lightPosition = v_500.allLights[i].lightPosition;
            arg.lightColor = v_500.allLights[i].lightColor;
            arg.lightDirection = v_500.allLights[i].lightDirection;
            arg.lightDistAttenCurveParams[0] = v_500.allLights[i].lightDistAttenCurveParams[0];
            arg.lightDistAttenCurveParams[1] = v_500.allLights[i].lightDistAttenCurveParams[1];
            arg.lightAngleAttenCurveParams[0] = v_500.allLights[i].lightAngleAttenCurveParams[0];
            arg.lightAngleAttenCurveParams[1] = v_500.allLights[i].lightAngleAttenCurveParams[1];
            arg.lightVP = v_500.allLights[i].lightVP;
            arg.padding[0] = v_500.allLights[i].padding[0];
            arg.padding[1] = v_500.allLights[i].padding[1];
            linearColor += apply_areaLight(arg, in.normal, v_500, in.v, diffuseMap, diffuseMapSmplr, in.uv, u_pushConstants);
        }
        else
        {
            Light arg_1;
            arg_1.lightType = v_500.allLights[i].lightType;
            arg_1.lightIntensity = v_500.allLights[i].lightIntensity;
            arg_1.lightCastShadow = v_500.allLights[i].lightCastShadow;
            arg_1.lightShadowMapIndex = v_500.allLights[i].lightShadowMapIndex;
            arg_1.lightAngleAttenCurveType = v_500.allLights[i].lightAngleAttenCurveType;
            arg_1.lightDistAttenCurveType = v_500.allLights[i].lightDistAttenCurveType;
            arg_1.lightSize = v_500.allLights[i].lightSize;
            arg_1.lightGUID = v_500.allLights[i].lightGUID;
            arg_1.lightPosition = v_500.allLights[i].lightPosition;
            arg_1.lightColor = v_500.allLights[i].lightColor;
            arg_1.lightDirection = v_500.allLights[i].lightDirection;
            arg_1.lightDistAttenCurveParams[0] = v_500.allLights[i].lightDistAttenCurveParams[0];
            arg_1.lightDistAttenCurveParams[1] = v_500.allLights[i].lightDistAttenCurveParams[1];
            arg_1.lightAngleAttenCurveParams[0] = v_500.allLights[i].lightAngleAttenCurveParams[0];
            arg_1.lightAngleAttenCurveParams[1] = v_500.allLights[i].lightAngleAttenCurveParams[1];
            arg_1.lightVP = v_500.allLights[i].lightVP;
            arg_1.padding[0] = v_500.allLights[i].padding[0];
            arg_1.padding[1] = v_500.allLights[i].padding[1];
            linearColor += apply_light(arg_1, cubeShadowMap, cubeShadowMapSmplr, shadowMap, shadowMapSmplr, globalShadowMap, globalShadowMapSmplr, in.normal, v_500, in.v, in.v_world, diffuseMap, diffuseMapSmplr, in.uv, u_pushConstants);
        }
    }
    linearColor += (skybox.sample(skyboxSmplr, float4(in.normal_world.xyz, 0.0).xyz, uint(round(float4(in.normal_world.xyz, 0.0).w)), level(8.0)).xyz * float3(0.20000000298023223876953125));
    float3 param = linearColor;
    linearColor = exposure_tone_mapping(param);
    float3 param_1 = linearColor;
    out.outputColor = float4(gamma_correction(param_1), 1.0);
    return out;
}

