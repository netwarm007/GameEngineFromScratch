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

static const float2 _354[4] = { float2(-0.94201624393463134765625f, -0.39906215667724609375f), float2(0.94558608531951904296875f, -0.768907248973846435546875f), float2(-0.094184100627899169921875f, -0.929388701915740966796875f), float2(0.34495937824249267578125f, 0.29387760162353515625f) };

cbuffer _500 : register(b0, space0)
{
    row_major float4x4 _500_viewMatrix : packoffset(c0);
    row_major float4x4 _500_projectionMatrix : packoffset(c4);
    float4 _500_camPos : packoffset(c8);
    int _500_numLights : packoffset(c9);
    Light _500_allLights[100] : packoffset(c10);
};
TextureCubeArray<float4> cubeShadowMap : register(t3, space0);
SamplerState _cubeShadowMap_sampler : register(s3, space0);
Texture2DArray<float4> shadowMap : register(t1, space0);
SamplerState _shadowMap_sampler : register(s1, space0);
Texture2DArray<float4> globalShadowMap : register(t2, space0);
SamplerState _globalShadowMap_sampler : register(s2, space0);
Texture2D<float4> diffuseMap : register(t0, space0);
SamplerState _diffuseMap_sampler : register(s0, space0);
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

static float4 normal;
static float4 v;
static float4 v_world;
static float2 uv;
static float4 normal_world;
static float4 outputColor;

struct SPIRV_Cross_Input
{
    float4 normal : TEXCOORD0;
    float4 normal_world : TEXCOORD1;
    float4 v : TEXCOORD2;
    float4 v_world : TEXCOORD3;
    float2 uv : TEXCOORD4;
};

struct SPIRV_Cross_Output
{
    float4 outputColor : SV_Target0;
};

float _131;

float3 projectOnPlane(float3 _point, float3 center_of_plane, float3 normal_of_plane)
{
    return _point - (normal_of_plane * dot(_point - center_of_plane, normal_of_plane));
}

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

bool isAbovePlane(float3 _point, float3 center_of_plane, float3 normal_of_plane)
{
    return dot(_point - center_of_plane, normal_of_plane) > 0.0f;
}

float3 linePlaneIntersect(float3 line_start, float3 line_dir, float3 center_of_plane, float3 normal_of_plane)
{
    return line_start + (line_dir * (dot(center_of_plane - line_start, normal_of_plane) / dot(line_dir, normal_of_plane)));
}

float3 apply_areaLight(Light light)
{
    float3 N = normalize(normal.xyz);
    float3 right = normalize(mul(float4(1.0f, 0.0f, 0.0f, 0.0f), _500_viewMatrix).xyz);
    float3 pnormal = normalize(mul(light.lightDirection, _500_viewMatrix).xyz);
    float3 ppos = mul(light.lightPosition, _500_viewMatrix).xyz;
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
    float4 param_5[2] = light.lightDistAttenCurveParams;
    float atten = apply_atten_curve(param_3, param_4, param_5);
    float3 linearColor = 0.0f.xxx;
    float pnDotL = dot(pnormal, -L);
    float nDotL = dot(N, L);
    float _741 = nDotL;
    bool _742 = _741 > 0.0f;
    bool _753;
    if (_742)
    {
        float3 param_6 = v.xyz;
        float3 param_7 = ppos;
        float3 param_8 = pnormal;
        _753 = isAbovePlane(param_6, param_7, param_8);
    }
    else
    {
        _753 = _742;
    }
    if (_753)
    {
        float3 V = normalize(-v.xyz);
        float3 R = normalize((N * (2.0f * dot(V, N))) - V);
        float3 R2 = normalize((N * (2.0f * dot(L, N))) - L);
        float3 param_9 = v.xyz;
        float3 param_10 = R;
        float3 param_11 = ppos;
        float3 param_12 = pnormal;
        float3 E = linePlaneIntersect(param_9, param_10, param_11, param_12);
        float specAngle = clamp(dot(-R, pnormal), 0.0f, 1.0f);
        float3 dirSpec = E - ppos;
        float2 dirSpec2D = float2(dot(dirSpec, right), dot(dirSpec, up));
        float2 nearestSpec2D = float2(clamp(dirSpec2D.x, -width, width), clamp(dirSpec2D.y, -height, height));
        float specFactor = 1.0f - clamp(length(nearestSpec2D - dirSpec2D), 0.0f, 1.0f);
        float3 admit_light = light.lightColor.xyz * (light.lightIntensity * atten);
        linearColor = (diffuseMap.Sample(_diffuseMap_sampler, uv).xyz * nDotL) * pnDotL;
        linearColor += (((0.800000011920928955078125f.xxx * pow(clamp(dot(R2, V), 0.0f, 1.0f), 50.0f)) * specFactor) * specAngle);
        linearColor *= admit_light;
    }
    return linearColor;
}

float shadow_test(float4 p, Light light, float cosTheta)
{
    float4 v_light_space = mul(p, light.lightVP);
    v_light_space /= v_light_space.w.xxxx;
    float visibility = 1.0f;
    if (light.lightShadowMapIndex != (-1))
    {
        float bias = 0.0005000000237487256526947021484375f * tan(acos(cosTheta));
        bias = clamp(bias, 0.0f, 0.00999999977648258209228515625f);
        float near_occ;
        switch (light.lightType)
        {
            case 0:
            {
                float3 L = p.xyz - light.lightPosition.xyz;
                near_occ = cubeShadowMap.Sample(_cubeShadowMap_sampler, float4(L, float(light.lightShadowMapIndex))).x;
                if ((length(L) - (near_occ * 10.0f)) > bias)
                {
                    visibility -= 0.87999999523162841796875f;
                }
                break;
            }
            case 1:
            {
                v_light_space = mul(v_light_space, float4x4(float4(0.5f, 0.0f, 0.0f, 0.0f), float4(0.0f, 0.5f, 0.0f, 0.0f), float4(0.0f, 0.0f, 0.5f, 0.0f), float4(0.5f, 0.5f, 0.5f, 1.0f)));
                for (int i = 0; i < 4; i++)
                {
                    float2 indexable[4] = _354;
                    near_occ = shadowMap.Sample(_shadowMap_sampler, float3(v_light_space.xy + (indexable[i] / 700.0f.xx), float(light.lightShadowMapIndex))).x;
                    if ((v_light_space.z - near_occ) > bias)
                    {
                        visibility -= 0.2199999988079071044921875f;
                    }
                }
                break;
            }
            case 2:
            {
                v_light_space = mul(v_light_space, float4x4(float4(0.5f, 0.0f, 0.0f, 0.0f), float4(0.0f, 0.5f, 0.0f, 0.0f), float4(0.0f, 0.0f, 0.5f, 0.0f), float4(0.5f, 0.5f, 0.5f, 1.0f)));
                for (int i_1 = 0; i_1 < 4; i_1++)
                {
                    float2 indexable_1[4] = _354;
                    near_occ = globalShadowMap.Sample(_globalShadowMap_sampler, float3(v_light_space.xy + (indexable_1[i_1] / 700.0f.xx), float(light.lightShadowMapIndex))).x;
                    if ((v_light_space.z - near_occ) > bias)
                    {
                        visibility -= 0.2199999988079071044921875f;
                    }
                }
                break;
            }
            case 3:
            {
                v_light_space = mul(v_light_space, float4x4(float4(0.5f, 0.0f, 0.0f, 0.0f), float4(0.0f, 0.5f, 0.0f, 0.0f), float4(0.0f, 0.0f, 0.5f, 0.0f), float4(0.5f, 0.5f, 0.5f, 1.0f)));
                for (int i_2 = 0; i_2 < 4; i_2++)
                {
                    float2 indexable_2[4] = _354;
                    near_occ = shadowMap.Sample(_shadowMap_sampler, float3(v_light_space.xy + (indexable_2[i_2] / 700.0f.xx), float(light.lightShadowMapIndex))).x;
                    if ((v_light_space.z - near_occ) > bias)
                    {
                        visibility -= 0.2199999988079071044921875f;
                    }
                }
                break;
            }
        }
    }
    return visibility;
}

float3 apply_light(Light light)
{
    float3 N = normalize(normal.xyz);
    float3 light_dir = normalize(mul(light.lightDirection, _500_viewMatrix).xyz);
    float3 L;
    if (light.lightPosition.w == 0.0f)
    {
        L = -light_dir;
    }
    else
    {
        L = mul(light.lightPosition, _500_viewMatrix).xyz - v.xyz;
    }
    float lightToSurfDist = length(L);
    L = normalize(L);
    float cosTheta = clamp(dot(N, L), 0.0f, 1.0f);
    float visibility = shadow_test(v_world, light, cosTheta);
    float lightToSurfAngle = acos(dot(L, -light_dir));
    float param = lightToSurfAngle;
    int param_1 = light.lightAngleAttenCurveType;
    float4 param_2[2] = light.lightAngleAttenCurveParams;
    float atten = apply_atten_curve(param, param_1, param_2);
    float param_3 = lightToSurfDist;
    int param_4 = light.lightDistAttenCurveType;
    float4 param_5[2] = light.lightDistAttenCurveParams;
    atten *= apply_atten_curve(param_3, param_4, param_5);
    float3 R = normalize((N * (2.0f * dot(L, N))) - L);
    float3 V = normalize(-v.xyz);
    float3 admit_light = light.lightColor.xyz * (light.lightIntensity * atten);
    float3 linearColor = diffuseMap.Sample(_diffuseMap_sampler, uv).xyz * cosTheta;
    if (visibility > 0.20000000298023223876953125f)
    {
        linearColor += (0.800000011920928955078125f.xxx * pow(clamp(dot(R, V), 0.0f, 1.0f), 50.0f));
    }
    linearColor *= admit_light;
    return linearColor * visibility;
}

float3 exposure_tone_mapping(float3 color)
{
    return 1.0f.xxx - exp((-color) * 1.0f);
}

float3 gamma_correction(float3 color)
{
    return pow(color, 0.4545454680919647216796875f.xxx);
}

void frag_main()
{
    float3 linearColor = 0.0f.xxx;
    for (int i = 0; i < _500_numLights; i++)
    {
        if (_500_allLights[i].lightType == 3)
        {
            Light arg;
            arg.lightIntensity = _500_allLights[i].lightIntensity;
            arg.lightType = _500_allLights[i].lightType;
            arg.lightCastShadow = _500_allLights[i].lightCastShadow;
            arg.lightShadowMapIndex = _500_allLights[i].lightShadowMapIndex;
            arg.lightAngleAttenCurveType = _500_allLights[i].lightAngleAttenCurveType;
            arg.lightDistAttenCurveType = _500_allLights[i].lightDistAttenCurveType;
            arg.lightSize = _500_allLights[i].lightSize;
            arg.lightGUID = _500_allLights[i].lightGUID;
            arg.lightPosition = _500_allLights[i].lightPosition;
            arg.lightColor = _500_allLights[i].lightColor;
            arg.lightDirection = _500_allLights[i].lightDirection;
            arg.lightDistAttenCurveParams[0] = _500_allLights[i].lightDistAttenCurveParams[0];
            arg.lightDistAttenCurveParams[1] = _500_allLights[i].lightDistAttenCurveParams[1];
            arg.lightAngleAttenCurveParams[0] = _500_allLights[i].lightAngleAttenCurveParams[0];
            arg.lightAngleAttenCurveParams[1] = _500_allLights[i].lightAngleAttenCurveParams[1];
            arg.lightVP = _500_allLights[i].lightVP;
            arg.padding[0] = _500_allLights[i].padding[0];
            arg.padding[1] = _500_allLights[i].padding[1];
            linearColor += apply_areaLight(arg);
        }
        else
        {
            Light arg_1;
            arg_1.lightIntensity = _500_allLights[i].lightIntensity;
            arg_1.lightType = _500_allLights[i].lightType;
            arg_1.lightCastShadow = _500_allLights[i].lightCastShadow;
            arg_1.lightShadowMapIndex = _500_allLights[i].lightShadowMapIndex;
            arg_1.lightAngleAttenCurveType = _500_allLights[i].lightAngleAttenCurveType;
            arg_1.lightDistAttenCurveType = _500_allLights[i].lightDistAttenCurveType;
            arg_1.lightSize = _500_allLights[i].lightSize;
            arg_1.lightGUID = _500_allLights[i].lightGUID;
            arg_1.lightPosition = _500_allLights[i].lightPosition;
            arg_1.lightColor = _500_allLights[i].lightColor;
            arg_1.lightDirection = _500_allLights[i].lightDirection;
            arg_1.lightDistAttenCurveParams[0] = _500_allLights[i].lightDistAttenCurveParams[0];
            arg_1.lightDistAttenCurveParams[1] = _500_allLights[i].lightDistAttenCurveParams[1];
            arg_1.lightAngleAttenCurveParams[0] = _500_allLights[i].lightAngleAttenCurveParams[0];
            arg_1.lightAngleAttenCurveParams[1] = _500_allLights[i].lightAngleAttenCurveParams[1];
            arg_1.lightVP = _500_allLights[i].lightVP;
            arg_1.padding[0] = _500_allLights[i].padding[0];
            arg_1.padding[1] = _500_allLights[i].padding[1];
            linearColor += apply_light(arg_1);
        }
    }
    linearColor += (skybox.SampleLevel(_skybox_sampler, float4(normal_world.xyz, 0.0f), 8.0f).xyz * 0.20000000298023223876953125f.xxx);
    float3 param = linearColor;
    linearColor = exposure_tone_mapping(param);
    float3 param_1 = linearColor;
    outputColor = float4(gamma_correction(param_1), 1.0f);
}

SPIRV_Cross_Output main(SPIRV_Cross_Input stage_input)
{
    normal = stage_input.normal;
    v = stage_input.v;
    v_world = stage_input.v_world;
    uv = stage_input.uv;
    normal_world = stage_input.normal_world;
    frag_main();
    SPIRV_Cross_Output stage_output;
    stage_output.outputColor = outputColor;
    return stage_output;
}
