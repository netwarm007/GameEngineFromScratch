#pragma clang diagnostic ignored "-Wmissing-prototypes"

#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct pbr_vert_output
{
    float4 pos;
    float4 normal;
    float4 normal_world;
    float4 v;
    float4 v_world;
    float2 uv;
    float3x3 TBN;
    float3 v_tangent;
    float3 camPos_tangent;
};

struct PerFrameConstants
{
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
    float4x4 arbitraryMatrix;
    float4 camPos;
    uint numLights;
};

struct Light
{
    float lightIntensity;
    uint lightType;
    int lightCastShadow;
    int lightShadowMapIndex;
    uint lightAngleAttenCurveType;
    uint lightDistAttenCurveType;
    float2 lightSize;
    uint4 lightGuid;
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
    float lightIntensity;
    uint lightType;
    int lightCastShadow;
    int lightShadowMapIndex;
    uint lightAngleAttenCurveType;
    uint lightDistAttenCurveType;
    float2 lightSize;
    uint4 lightGuid;
    float4 lightPosition;
    float4 lightColor;
    float4 lightDirection;
    float4 lightDistAttenCurveParams[2];
    float4 lightAngleAttenCurveParams[2];
    float4x4 lightVP;
    float4 padding[2];
};

struct LightInfo
{
    Light_1 lights[100];
};

struct PerBatchConstants
{
    float4x4 modelMatrix;
};

constant float _101 = {};

struct pbr_frag_main_out
{
    float4 _entryPointOutput [[color(0)]];
};

struct pbr_frag_main_in
{
    float4 input_normal [[user(locn0)]];
    float4 input_normal_world [[user(locn1)]];
    float4 input_v [[user(locn2)]];
    float4 input_v_world [[user(locn3)]];
    float2 input_uv [[user(locn4)]];
    float3 input_TBN_0 [[user(locn5)]];
    float3 input_TBN_1 [[user(locn6)]];
    float3 input_TBN_2 [[user(locn7)]];
    float3 input_v_tangent [[user(locn8)]];
    float3 input_camPos_tangent [[user(locn9)]];
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

float3 inverse_gamma_correction(thread const float3& color)
{
    return pow(max(color, float3(0.0)), float3(2.2000000476837158203125));
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
            float param_3 = param;
            float param_4 = param_1;
            float param_5 = param_2;
            atten = linear_interpolate(param_3, param_4, param_5);
            break;
        }
        case 2:
        {
            float begin_atten_1 = atten_params[0].x;
            float end_atten_1 = atten_params[0].y;
            float param_3_1 = dist;
            float param_4_1 = begin_atten_1;
            float param_5_1 = end_atten_1;
            float param_6 = param_3_1;
            float param_7 = param_4_1;
            float param_8 = param_5_1;
            float tmp = linear_interpolate(param_6, param_7, param_8);
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

float DistributionGGX(thread const float3& N, thread const float3& H, thread const float& roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float num = a2;
    float denom = (NdotH2 * (a2 - 1.0)) + 1.0;
    denom = (3.1415927410125732421875 * denom) * denom;
    return num / denom;
}

float GeometrySchlickGGXDirect(thread const float& NdotV, thread const float& roughness)
{
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    float num = NdotV;
    float denom = (NdotV * (1.0 - k)) + k;
    return num / denom;
}

float GeometrySmithDirect(thread const float3& N, thread const float3& V, thread const float3& L, thread const float& roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float param = NdotV;
    float param_1 = roughness;
    float ggx2 = GeometrySchlickGGXDirect(param, param_1);
    float param_2 = NdotL;
    float param_3 = roughness;
    float ggx1 = GeometrySchlickGGXDirect(param_2, param_3);
    return ggx1 * ggx2;
}

float3 fresnelSchlick(thread const float& cosTheta, thread const float3& F0)
{
    return F0 + ((float3(1.0) - F0) * pow(1.0 - cosTheta, 5.0));
}

float3 fresnelSchlickRoughness(thread const float& cosTheta, thread const float3& F0, thread const float& roughness)
{
    return F0 + ((max(float3(1.0 - roughness), F0) - F0) * pow(1.0 - cosTheta, 5.0));
}

float3 reinhard_tone_mapping(thread const float3& color)
{
    return color / (color + float3(1.0));
}

float3 gamma_correction(thread const float3& color)
{
    return pow(max(color, float3(0.0)), float3(0.4545454680919647216796875));
}

float4 _pbr_frag_main(thread const pbr_vert_output& _input, thread texture2d<float> normalMap, thread sampler samp0, constant PerFrameConstants& v_402, thread texture2d<float> diffuseMap, thread texture2d<float> metallicMap, thread texture2d<float> roughnessMap, constant LightInfo& v_479, thread texture2d<float> aoMap, thread texturecube_array<float> skybox, thread texture2d<float> brdfLUT)
{
    float2 texCoords = _input.uv;
    float3 tangent_normal = normalMap.sample(samp0, texCoords).xyz;
    tangent_normal = (tangent_normal * 2.0) - float3(1.0);
    float3 N = normalize(_input.TBN * tangent_normal);
    float3 V = normalize(v_402.camPos.xyz - _input.v_world.xyz);
    float3 R = reflect(-V, N);
    float3 param = diffuseMap.sample(samp0, texCoords).xyz;
    float3 albedo = inverse_gamma_correction(param);
    float meta = metallicMap.sample(samp0, texCoords).x;
    float rough = roughnessMap.sample(samp0, texCoords).x;
    float3 F0 = float3(0.039999999105930328369140625);
    F0 = mix(F0, albedo, float3(meta));
    float3 Lo = float3(0.0);
    for (int i = 0; uint(i) < v_402.numLights; i++)
    {
        Light light;
        light.lightIntensity = v_479.lights[i].lightIntensity;
        light.lightType = v_479.lights[i].lightType;
        light.lightCastShadow = v_479.lights[i].lightCastShadow;
        light.lightShadowMapIndex = v_479.lights[i].lightShadowMapIndex;
        light.lightAngleAttenCurveType = v_479.lights[i].lightAngleAttenCurveType;
        light.lightDistAttenCurveType = v_479.lights[i].lightDistAttenCurveType;
        light.lightSize = v_479.lights[i].lightSize;
        light.lightGuid = v_479.lights[i].lightGuid;
        light.lightPosition = v_479.lights[i].lightPosition;
        light.lightColor = v_479.lights[i].lightColor;
        light.lightDirection = v_479.lights[i].lightDirection;
        light.lightDistAttenCurveParams[0] = v_479.lights[i].lightDistAttenCurveParams[0];
        light.lightDistAttenCurveParams[1] = v_479.lights[i].lightDistAttenCurveParams[1];
        light.lightAngleAttenCurveParams[0] = v_479.lights[i].lightAngleAttenCurveParams[0];
        light.lightAngleAttenCurveParams[1] = v_479.lights[i].lightAngleAttenCurveParams[1];
        light.lightVP = v_479.lights[i].lightVP;
        light.padding[0] = v_479.lights[i].padding[0];
        light.padding[1] = v_479.lights[i].padding[1];
        float3 L = normalize(light.lightPosition.xyz - _input.v_world.xyz);
        float3 H = normalize(V + L);
        float NdotL = max(dot(N, L), 0.0);
        float visibility = 1.0;
        float lightToSurfDist = length(L);
        float lightToSurfAngle = acos(dot(-L, light.lightDirection.xyz));
        float param_1 = lightToSurfAngle;
        int param_2 = int(light.lightAngleAttenCurveType);
        float4 param_3[2];
        spvArrayCopy(param_3, light.lightAngleAttenCurveParams);
        float atten = apply_atten_curve(param_1, param_2, param_3);
        float param_4 = lightToSurfDist;
        int param_5 = int(light.lightDistAttenCurveType);
        float4 param_6[2];
        spvArrayCopy(param_6, light.lightDistAttenCurveParams);
        atten *= apply_atten_curve(param_4, param_5, param_6);
        float3 radiance = light.lightColor.xyz * (light.lightIntensity * atten);
        float3 param_7 = N;
        float3 param_8 = H;
        float param_9 = rough;
        float NDF = DistributionGGX(param_7, param_8, param_9);
        float3 param_10 = N;
        float3 param_11 = V;
        float3 param_12 = L;
        float param_13 = rough;
        float G = GeometrySmithDirect(param_10, param_11, param_12, param_13);
        float param_14 = max(dot(H, V), 0.0);
        float3 param_15 = F0;
        float3 F = fresnelSchlick(param_14, param_15);
        float3 kS = F;
        float3 kD = float3(1.0) - kS;
        kD *= (1.0 - meta);
        float3 numerator = F * (NDF * G);
        float denominator = (4.0 * max(dot(N, V), 0.0)) * NdotL;
        float3 specular = numerator / float3(max(denominator, 0.001000000047497451305389404296875));
        Lo += ((((((kD * albedo) / float3(3.1415927410125732421875)) + specular) * radiance) * NdotL) * visibility);
    }
    float ambientOcc = aoMap.sample(samp0, texCoords).x;
    float param_16 = max(dot(N, V), 0.0);
    float3 param_17 = F0;
    float param_18 = rough;
    float3 F_1 = fresnelSchlickRoughness(param_16, param_17, param_18);
    float3 kS_1 = F_1;
    float3 kD_1 = float3(1.0) - kS_1;
    kD_1 *= (1.0 - meta);
    float3 irradiance = skybox.sample(samp0, float4(N, 0.0).xyz, uint(round(float4(N, 0.0).w)), level(1.0)).xyz;
    float3 diffuse = irradiance * albedo;
    float3 prefilteredColor = skybox.sample(samp0, float4(R, 1.0).xyz, uint(round(float4(R, 1.0).w)), level(rough * 9.0)).xyz;
    float2 envBRDF = brdfLUT.sample(samp0, float2(max(dot(N, V), 0.0), rough)).xy;
    float3 specular_1 = prefilteredColor * ((F_1 * envBRDF.x) + float3(envBRDF.y));
    float3 ambient = ((kD_1 * diffuse) + specular_1) * ambientOcc;
    float3 linearColor = ambient + Lo;
    float3 param_19 = linearColor;
    linearColor = reinhard_tone_mapping(param_19);
    float3 param_20 = linearColor;
    linearColor = gamma_correction(param_20);
    return float4(linearColor, 1.0);
}

fragment pbr_frag_main_out pbr_frag_main(pbr_frag_main_in in [[stage_in]], constant PerFrameConstants& v_402 [[buffer(10)]], constant LightInfo& v_479 [[buffer(12)]], texture2d<float> diffuseMap [[texture(0)]], texture2d<float> normalMap [[texture(1)]], texture2d<float> metallicMap [[texture(2)]], texture2d<float> roughnessMap [[texture(3)]], texture2d<float> aoMap [[texture(4)]], texture2d<float> brdfLUT [[texture(6)]], texturecube_array<float> skybox [[texture(10)]], sampler samp0 [[sampler(0)]], float4 gl_FragCoord [[position]])
{
    pbr_frag_main_out out = {};
    float3x3 input_TBN = {};
    input_TBN[0] = in.input_TBN_0;
    input_TBN[1] = in.input_TBN_1;
    input_TBN[2] = in.input_TBN_2;
    pbr_vert_output _input;
    _input.pos = gl_FragCoord;
    _input.normal = in.input_normal;
    _input.normal_world = in.input_normal_world;
    _input.v = in.input_v;
    _input.v_world = in.input_v_world;
    _input.uv = in.input_uv;
    _input.TBN = input_TBN;
    _input.v_tangent = in.input_v_tangent;
    _input.camPos_tangent = in.input_camPos_tangent;
    pbr_vert_output param = _input;
    out._entryPointOutput = _pbr_frag_main(param, normalMap, samp0, v_402, diffuseMap, metallicMap, roughnessMap, v_479, aoMap, skybox, brdfLUT);
    return out;
}

