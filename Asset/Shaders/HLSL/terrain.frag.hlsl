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
    row_major float4x4 lightVP;
    float4 padding[2];
};

cbuffer _193 : register(b0, space0)
{
    row_major float4x4 _193_viewMatrix : packoffset(c0);
    row_major float4x4 _193_projectionMatrix : packoffset(c4);
    float4 _193_camPos : packoffset(c8);
    int _193_numLights : packoffset(c9);
    Light _193_allLights[100] : packoffset(c10);
};
Texture2D<float4> diffuseMap : register(t0, space0);
SamplerState _diffuseMap_sampler : register(s0, space0);
Texture2DArray<float4> shadowMap : register(t1, space0);
SamplerState _shadowMap_sampler : register(s1, space0);
Texture2DArray<float4> globalShadowMap : register(t2, space0);
SamplerState _globalShadowMap_sampler : register(s2, space0);
TextureCubeArray<float4> cubeShadowMap : register(t3, space0);
SamplerState _cubeShadowMap_sampler : register(s3, space0);
TextureCubeArray<float4> skybox : register(t4, space0);
SamplerState _skybox_sampler : register(s4, space0);
Texture2D<float4> normalMap : register(t5, space0);
SamplerState _normalMap_sampler : register(s5, space0);
Texture2D<float4> metallicMap : register(t6, space0);
SamplerState _metallicMap_sampler : register(s6, space0);
Texture2D<float4> roughnessMap : register(t7, space0);
SamplerState _roughnessMap_sampler : register(s7, space0);
Texture2D<float4> aoMap : register(t8, space0);
SamplerState _aoMap_sampler : register(s8, space0);
Texture2D<float4> brdfLUT : register(t9, space0);
SamplerState _brdfLUT_sampler : register(s9, space0);
Texture2D<float4> heightMap : register(t10, space0);
SamplerState _heightMap_sampler : register(s10, space0);
Texture2D<float4> terrainHeightMap : register(t11, space0);
SamplerState _terrainHeightMap_sampler : register(s11, space0);

static float4 v_world;
static float4 normal_world;
static float4 outputColor;
static float2 uv;
static float3x3 TBN;
static float3 v_tangent;
static float3 camPos_tangent;

struct SPIRV_Cross_Input
{
    float4 normal_world : TEXCOORD1;
    float4 v_world : TEXCOORD3;
    float2 uv : TEXCOORD4;
    float3x3 TBN : TEXCOORD5;
    float3 v_tangent : TEXCOORD8;
    float3 camPos_tangent : TEXCOORD9;
};

struct SPIRV_Cross_Output
{
    float4 outputColor : SV_Target0;
};

float _52;

float linear_interpolate(float t, float begin, float end)
{
    if (t < begin)
    {
        return 1.0f;
    }
    else
    {
        if (t > end)
        {
            return 0.0f;
        }
        else
        {
            return (end - t) / (end - begin);
        }
    }
}

float apply_atten_curve(float dist, int atten_curve_type, float4 atten_params[2])
{
    float atten = 1.0f;
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
            atten = (3.0f * pow(tmp, 2.0f)) - (2.0f * pow(tmp, 3.0f));
            break;
        }
        case 3:
        {
            float scale = atten_params[0].x;
            float offset = atten_params[0].y;
            float kl = atten_params[0].z;
            float kc = atten_params[0].w;
            atten = clamp((scale / ((kl * dist) + (kc * scale))) + offset, 0.0f, 1.0f);
            break;
        }
        case 4:
        {
            float scale_1 = atten_params[0].x;
            float offset_1 = atten_params[0].y;
            float kq = atten_params[0].z;
            float kl_1 = atten_params[0].w;
            float kc_1 = atten_params[1].x;
            atten = clamp(pow(scale_1, 2.0f) / ((((kq * pow(dist, 2.0f)) + ((kl_1 * dist) * scale_1)) + (kc_1 * pow(scale_1, 2.0f))) + offset_1), 0.0f, 1.0f);
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

void frag_main()
{
    float3 Lo = 0.0f.xxx;
    for (int i = 0; i < _193_numLights; i++)
    {
        Light light;
        light.lightIntensity = _193_allLights[i].lightIntensity;
        light.lightType = _193_allLights[i].lightType;
        light.lightCastShadow = _193_allLights[i].lightCastShadow;
        light.lightShadowMapIndex = _193_allLights[i].lightShadowMapIndex;
        light.lightAngleAttenCurveType = _193_allLights[i].lightAngleAttenCurveType;
        light.lightDistAttenCurveType = _193_allLights[i].lightDistAttenCurveType;
        light.lightSize = _193_allLights[i].lightSize;
        light.lightGUID = _193_allLights[i].lightGUID;
        light.lightPosition = _193_allLights[i].lightPosition;
        light.lightColor = _193_allLights[i].lightColor;
        light.lightDirection = _193_allLights[i].lightDirection;
        light.lightDistAttenCurveParams[0] = _193_allLights[i].lightDistAttenCurveParams[0];
        light.lightDistAttenCurveParams[1] = _193_allLights[i].lightDistAttenCurveParams[1];
        light.lightAngleAttenCurveParams[0] = _193_allLights[i].lightAngleAttenCurveParams[0];
        light.lightAngleAttenCurveParams[1] = _193_allLights[i].lightAngleAttenCurveParams[1];
        light.lightVP = _193_allLights[i].lightVP;
        light.padding[0] = _193_allLights[i].padding[0];
        light.padding[1] = _193_allLights[i].padding[1];
        float3 L = normalize(light.lightPosition.xyz - v_world.xyz);
        float3 N = normal_world.xyz;
        float NdotL = max(dot(N, L), 0.0f);
        float visibility = 1.0f;
        float lightToSurfDist = length(L);
        float lightToSurfAngle = acos(dot(-L, light.lightDirection.xyz));
        float param = lightToSurfAngle;
        int param_1 = light.lightAngleAttenCurveType;
        float4 param_2[2] = light.lightAngleAttenCurveParams;
        float atten = apply_atten_curve(param, param_1, param_2);
        float param_3 = lightToSurfDist;
        int param_4 = light.lightDistAttenCurveType;
        float4 param_5[2] = light.lightDistAttenCurveParams;
        atten *= apply_atten_curve(param_3, param_4, param_5);
        float3 radiance = light.lightColor.xyz * (light.lightIntensity * atten);
        Lo += ((radiance * NdotL) * visibility);
    }
    outputColor = float4(Lo, 1.0f);
}

SPIRV_Cross_Output terrain_frag_main(SPIRV_Cross_Input _entryPointOutput)
{
    v_world = _entryPointOutput.v_world;
    normal_world = _entryPointOutput.normal_world;
    uv = _entryPointOutput.uv;
    TBN = _entryPointOutput.TBN;
    v_tangent = _entryPointOutput.v_tangent;
    camPos_tangent = _entryPointOutput.camPos_tangent;
    frag_main();
    SPIRV_Cross_Output stage_output;
    stage_output.outputColor = outputColor;
    return stage_output;
}
