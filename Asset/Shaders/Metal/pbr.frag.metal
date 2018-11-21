#pragma clang diagnostic ignored "-Wmissing-prototypes"

#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

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

struct vert_output
{
    float4 position;
    float4 normal;
    float4 normal_world;
    float4 v;
    float4 v_world;
    float2 uv;
    float3x3 TBN;
    float3 v_tangent;
    float3 camPos_tangent;
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

struct PerFrameConstants
{
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
    float4 camPos;
    uint numLights;
    char pad4[12];
    float padding[3];
    Light_1 lights[100];
};

struct PerBatchConstants
{
    float4x4 modelMatrix;
};

constant float _118 = {};

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

float2 ParallaxMapping(thread const float2& texCoords, thread const float3& viewDir, thread sampler samp0, thread texture2d<float> heightMap)
{
    float param = abs(dot(float3(0.0, 0.0, 1.0), viewDir));
    float param_1 = 32.0;
    float param_2 = 8.0;
    float numLayers = linear_interpolate(param, param_1, param_2);
    float layerDepth = 1.0 / numLayers;
    float currentLayerDepth = 0.0;
    float2 currentTexCoords = texCoords;
    float currentDepthMapValue = 1.0 - heightMap.sample(samp0, currentTexCoords).x;
    float2 P = viewDir.xy * 0.100000001490116119384765625;
    float2 deltaTexCoords = P / float2(numLayers);
    while (currentLayerDepth < currentDepthMapValue)
    {
        currentTexCoords -= deltaTexCoords;
        currentDepthMapValue = 1.0 - heightMap.sample(samp0, currentTexCoords).x;
        currentLayerDepth += layerDepth;
    }
    float2 prevTexCoords = currentTexCoords + deltaTexCoords;
    float afterDepth = currentDepthMapValue - currentLayerDepth;
    float beforeDepth = ((1.0 - heightMap.sample(samp0, prevTexCoords).x) - currentLayerDepth) + layerDepth;
    float weight = afterDepth / (afterDepth - beforeDepth);
    float2 finalTexCoords = (prevTexCoords * weight) + (currentTexCoords * (1.0 - weight));
    return finalTexCoords;
}

float3 inverse_gamma_correction(thread const float3& color)
{
    return pow(max(color, float3(0.0)), float3(2.2000000476837158203125));
}

float shadow_test(thread const float4& p, thread const Light& light, thread const float& cosTheta, thread texturecube_array<float> cubeShadowMap, thread sampler samp0, thread texture2d_array<float> shadowMap, thread texture2d_array<float> globalShadowMap)
{
    float4 v_light_space = light.lightVP * p;
    v_light_space /= float4(v_light_space.w);
    float visibility = 1.0;
    if (light.lightShadowMapIndex != (-1))
    {
        float bias0 = 0.0005000000237487256526947021484375 * tan(acos(cosTheta));
        bias0 = clamp(bias0, 0.0, 0.00999999977648258209228515625);
        float near_occ;
        int i;
        switch (light.lightType)
        {
            case 0:
            {
                float3 L = p.xyz - light.lightPosition.xyz;
                near_occ = cubeShadowMap.sample(samp0, float4(L, float(light.lightShadowMapIndex)).xyz, uint(round(float4(L, float(light.lightShadowMapIndex)).w))).x;
                if ((length(L) - (near_occ * 10.0)) > bias0)
                {
                    visibility -= 0.87999999523162841796875;
                }
                break;
            }
            case 1:
            {
                v_light_space *= float4x4(float4(0.5, 0.0, 0.0, 0.0), float4(0.0, 0.5, 0.0, 0.0), float4(0.0, 0.0, 0.5, 0.0), float4(0.5, 0.5, 0.5, 1.0));
                i = 0;
                for (; i < 4; i++)
                {
                    float4x2 indexable = float4x2(float2(-0.94201624393463134765625, -0.39906215667724609375), float2(0.94558608531951904296875, -0.768907248973846435546875), float2(-0.094184100627899169921875, -0.929388701915740966796875), float2(0.34495937824249267578125, 0.29387760162353515625));
                    near_occ = shadowMap.sample(samp0, float3(v_light_space.xy + (indexable[i] / float2(700.0)), float(light.lightShadowMapIndex)).xy, uint(round(float3(v_light_space.xy + (indexable[i] / float2(700.0)), float(light.lightShadowMapIndex)).z))).x;
                    if ((v_light_space.z - near_occ) > bias0)
                    {
                        visibility -= 0.2199999988079071044921875;
                    }
                }
                break;
            }
            case 2:
            {
                v_light_space *= float4x4(float4(0.5, 0.0, 0.0, 0.0), float4(0.0, 0.5, 0.0, 0.0), float4(0.0, 0.0, 0.5, 0.0), float4(0.5, 0.5, 0.5, 1.0));
                i = 0;
                for (; i < 4; i++)
                {
                    float4x2 indexable_1 = float4x2(float2(-0.94201624393463134765625, -0.39906215667724609375), float2(0.94558608531951904296875, -0.768907248973846435546875), float2(-0.094184100627899169921875, -0.929388701915740966796875), float2(0.34495937824249267578125, 0.29387760162353515625));
                    near_occ = globalShadowMap.sample(samp0, float3(v_light_space.xy + (indexable_1[i] / float2(700.0)), float(light.lightShadowMapIndex)).xy, uint(round(float3(v_light_space.xy + (indexable_1[i] / float2(700.0)), float(light.lightShadowMapIndex)).z))).x;
                    if ((v_light_space.z - near_occ) > bias0)
                    {
                        visibility -= 0.2199999988079071044921875;
                    }
                }
                break;
            }
            case 3:
            {
                v_light_space *= float4x4(float4(0.5, 0.0, 0.0, 0.0), float4(0.0, 0.5, 0.0, 0.0), float4(0.0, 0.0, 0.5, 0.0), float4(0.5, 0.5, 0.5, 1.0));
                i = 0;
                for (; i < 4; i++)
                {
                    float4x2 indexable_2 = float4x2(float2(-0.94201624393463134765625, -0.39906215667724609375), float2(0.94558608531951904296875, -0.768907248973846435546875), float2(-0.094184100627899169921875, -0.929388701915740966796875), float2(0.34495937824249267578125, 0.29387760162353515625));
                    near_occ = shadowMap.sample(samp0, float3(v_light_space.xy + (indexable_2[i] / float2(700.0)), float(light.lightShadowMapIndex)).xy, uint(round(float3(v_light_space.xy + (indexable_2[i] / float2(700.0)), float(light.lightShadowMapIndex)).z))).x;
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

float4 _pbr_frag_main(thread const vert_output& _input, thread texturecube_array<float> cubeShadowMap, thread sampler samp0, thread texture2d_array<float> shadowMap, thread texture2d_array<float> globalShadowMap, thread texture2d<float> heightMap, thread texture2d<float> normalMap, constant PerFrameConstants& v_765, thread texture2d<float> diffuseMap, thread texture2d<float> metalicMap, thread texture2d<float> roughnessMap, thread texture2d<float> aoMap, thread texturecube_array<float> skybox, thread texture2d<float> brdfLUT)
{
    float3 viewDir = normalize(_input.camPos_tangent - _input.v_tangent);
    float2 param = _input.uv;
    float3 param_1 = viewDir;
    float2 texCoords = ParallaxMapping(param, param_1, samp0, heightMap);
    float3 tangent_normal = normalMap.sample(samp0, texCoords).xyz;
    tangent_normal = (tangent_normal * 2.0) - float3(1.0);
    float3 N = normalize(_input.TBN * tangent_normal);
    float3 V = normalize(v_765.camPos.xyz - _input.v_world.xyz);
    float3 R = reflect(-V, N);
    float3 param_2 = diffuseMap.sample(samp0, texCoords).xyz;
    float3 albedo = inverse_gamma_correction(param_2);
    float meta = metalicMap.sample(samp0, texCoords).x;
    float rough = roughnessMap.sample(samp0, texCoords).x;
    float3 F0 = float3(0.039999999105930328369140625);
    F0 = mix(F0, albedo, float3(meta));
    float3 Lo = float3(0.0);
    for (int i = 0; uint(i) < v_765.numLights; i++)
    {
        Light light;
        light.lightIntensity = v_765.lights[i].lightIntensity;
        light.lightType = v_765.lights[i].lightType;
        light.lightCastShadow = v_765.lights[i].lightCastShadow;
        light.lightShadowMapIndex = v_765.lights[i].lightShadowMapIndex;
        light.lightAngleAttenCurveType = v_765.lights[i].lightAngleAttenCurveType;
        light.lightDistAttenCurveType = v_765.lights[i].lightDistAttenCurveType;
        light.lightSize = v_765.lights[i].lightSize;
        light.lightGuid = v_765.lights[i].lightGuid;
        light.lightPosition = v_765.lights[i].lightPosition;
        light.lightColor = v_765.lights[i].lightColor;
        light.lightDirection = v_765.lights[i].lightDirection;
        light.lightDistAttenCurveParams[0] = v_765.lights[i].lightDistAttenCurveParams[0];
        light.lightDistAttenCurveParams[1] = v_765.lights[i].lightDistAttenCurveParams[1];
        light.lightAngleAttenCurveParams[0] = v_765.lights[i].lightAngleAttenCurveParams[0];
        light.lightAngleAttenCurveParams[1] = v_765.lights[i].lightAngleAttenCurveParams[1];
        light.lightVP = v_765.lights[i].lightVP;
        light.padding[0] = v_765.lights[i].padding[0];
        light.padding[1] = v_765.lights[i].padding[1];
        float3 L = normalize(light.lightPosition.xyz - _input.v_world.xyz);
        float3 H = normalize(V + L);
        float NdotL = max(dot(N, L), 0.0);
        float4 param_3 = _input.v_world;
        Light param_4 = light;
        float param_5 = NdotL;
        float visibility = shadow_test(param_3, param_4, param_5, cubeShadowMap, samp0, shadowMap, globalShadowMap);
        float lightToSurfDist = length(L);
        float lightToSurfAngle = acos(dot(-L, light.lightDirection.xyz));
        float param_6 = lightToSurfAngle;
        int param_7 = int(light.lightAngleAttenCurveType);
        float4 param_8[2];
        spvArrayCopy(param_8, light.lightAngleAttenCurveParams);
        float atten = apply_atten_curve(param_6, param_7, param_8);
        float param_9 = lightToSurfDist;
        int param_10 = int(light.lightDistAttenCurveType);
        float4 param_11[2];
        spvArrayCopy(param_11, light.lightDistAttenCurveParams);
        atten *= apply_atten_curve(param_9, param_10, param_11);
        float3 radiance = light.lightColor.xyz * (light.lightIntensity * atten);
        float3 param_12 = N;
        float3 param_13 = H;
        float param_14 = rough;
        float NDF = DistributionGGX(param_12, param_13, param_14);
        float3 param_15 = N;
        float3 param_16 = V;
        float3 param_17 = L;
        float param_18 = rough;
        float G = GeometrySmithDirect(param_15, param_16, param_17, param_18);
        float param_19 = max(dot(H, V), 0.0);
        float3 param_20 = F0;
        float3 F = fresnelSchlick(param_19, param_20);
        float3 kS = F;
        float3 kD = float3(1.0) - kS;
        kD *= (1.0 - meta);
        float3 numerator = F * (NDF * G);
        float denominator = (4.0 * max(dot(N, V), 0.0)) * NdotL;
        float3 specular = numerator / float3(max(denominator, 0.001000000047497451305389404296875));
        Lo += ((((((kD * albedo) / float3(3.1415927410125732421875)) + specular) * radiance) * NdotL) * visibility);
    }
    float ambientOcc = aoMap.sample(samp0, texCoords).x;
    float param_21 = max(dot(N, V), 0.0);
    float3 param_22 = F0;
    float param_23 = rough;
    float3 F_1 = fresnelSchlickRoughness(param_21, param_22, param_23);
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
    float3 param_24 = linearColor;
    linearColor = reinhard_tone_mapping(param_24);
    float3 param_25 = linearColor;
    linearColor = gamma_correction(param_25);
    return float4(linearColor, 1.0);
}

fragment pbr_frag_main_out pbr_frag_main(pbr_frag_main_in in [[stage_in]], constant PerFrameConstants& v_765 [[buffer(0)]], texture2d<float> diffuseMap [[texture(0)]], texture2d<float> normalMap [[texture(1)]], texture2d<float> metalicMap [[texture(2)]], texture2d<float> roughnessMap [[texture(3)]], texture2d<float> aoMap [[texture(4)]], texture2d<float> heightMap [[texture(5)]], texture2d<float> brdfLUT [[texture(6)]], texture2d_array<float> shadowMap [[texture(7)]], texture2d_array<float> globalShadowMap [[texture(8)]], texturecube_array<float> cubeShadowMap [[texture(9)]], texturecube_array<float> skybox [[texture(10)]], sampler samp0 [[sampler(0)]], float4 gl_FragCoord [[position]])
{
    pbr_frag_main_out out = {};
    float3x3 input_TBN = {};
    input_TBN[0] = in.input_TBN_0;
    input_TBN[1] = in.input_TBN_1;
    input_TBN[2] = in.input_TBN_2;
    vert_output _input;
    _input.position = gl_FragCoord;
    _input.normal = in.input_normal;
    _input.normal_world = in.input_normal_world;
    _input.v = in.input_v;
    _input.v_world = in.input_v_world;
    _input.uv = in.input_uv;
    _input.TBN = input_TBN;
    _input.v_tangent = in.input_v_tangent;
    _input.camPos_tangent = in.input_camPos_tangent;
    vert_output param = _input;
    out._entryPointOutput = _pbr_frag_main(param, cubeShadowMap, samp0, shadowMap, globalShadowMap, heightMap, normalMap, v_765, diffuseMap, metalicMap, roughnessMap, aoMap, skybox, brdfLUT);
    return out;
}

