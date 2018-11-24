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
    float4 camPos;
    uint numLights;
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

struct Light_2
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

struct PerBatchConstants
{
    float4x4 modelMatrix;
};

constant float _131 = {};

struct basic_frag_main_out
{
    float4 _entryPointOutput [[color(0)]];
};

struct basic_frag_main_in
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

float3 projectOnPlane(thread const float3& _point, thread const float3& center_of_plane, thread const float3& normal_of_plane)
{
    return _point - (normal_of_plane * dot(_point - center_of_plane, normal_of_plane));
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

bool isAbovePlane(thread const float3& _point, thread const float3& center_of_plane, thread const float3& normal_of_plane)
{
    return dot(_point - center_of_plane, normal_of_plane) > 0.0;
}

float3 linePlaneIntersect(thread const float3& line_start, thread const float3& line_dir, thread const float3& center_of_plane, thread const float3& normal_of_plane)
{
    return line_start + (line_dir * (dot(center_of_plane - line_start, normal_of_plane) / dot(line_dir, normal_of_plane)));
}

float3 apply_areaLight(thread const Light& light, thread const vert_output& _input, constant PerFrameConstants& v_281, thread texture2d<float> diffuseMap, thread sampler samp0)
{
    float3 N = normalize(_input.normal.xyz);
    float3 right = normalize((v_281.viewMatrix * float4(1.0, 0.0, 0.0, 0.0)).xyz);
    float3 pnormal = normalize((v_281.viewMatrix * light.lightDirection).xyz);
    float3 ppos = (v_281.viewMatrix * light.lightPosition).xyz;
    float3 up = normalize(cross(pnormal, right));
    right = normalize(cross(up, pnormal));
    float width = light.lightSize.x;
    float height = light.lightSize.y;
    float3 param = _input.v.xyz;
    float3 param_1 = ppos;
    float3 param_2 = pnormal;
    float3 projection = projectOnPlane(param, param_1, param_2);
    float3 dir = projection - ppos;
    float2 diagonal = float2(dot(dir, right), dot(dir, up));
    float2 nearest2D = float2(clamp(diagonal.x, -width, width), clamp(diagonal.y, -height, height));
    float3 nearestPointInside = (ppos + (right * nearest2D.x)) + (up * nearest2D.y);
    float3 L = nearestPointInside - _input.v.xyz;
    float lightToSurfDist = length(L);
    L = normalize(L);
    float param_3 = lightToSurfDist;
    int param_4 = int(light.lightDistAttenCurveType);
    float4 param_5[2];
    spvArrayCopy(param_5, light.lightDistAttenCurveParams);
    float atten = apply_atten_curve(param_3, param_4, param_5);
    float3 linearColor = float3(0.0);
    float pnDotL = dot(pnormal, -L);
    float nDotL = dot(N, L);
    float3 param_6 = _input.v.xyz;
    float3 param_7 = ppos;
    float3 param_8 = pnormal;
    if ((nDotL > 0.0) && isAbovePlane(param_6, param_7, param_8))
    {
        float3 V = normalize(-_input.v.xyz);
        float3 R = normalize((N * (2.0 * dot(V, N))) - V);
        float3 R2 = normalize((N * (2.0 * dot(L, N))) - L);
        float3 param_9 = _input.v.xyz;
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
        linearColor = (diffuseMap.sample(samp0, _input.uv).xyz * nDotL) * pnDotL;
        linearColor += (((float3(0.800000011920928955078125) * pow(clamp(dot(R2, V), 0.0, 1.0), 50.0)) * specFactor) * specAngle);
        linearColor *= admit_light;
    }
    return linearColor;
}

float3 apply_light(thread const Light& light, thread const vert_output& _input, constant PerFrameConstants& v_281, thread texture2d<float> diffuseMap, thread sampler samp0)
{
    float3 N = normalize(_input.normal.xyz);
    float3 light_dir = normalize((v_281.viewMatrix * light.lightDirection).xyz);
    float3 L;
    if (light.lightPosition.w == 0.0)
    {
        L = -light_dir;
    }
    else
    {
        L = (v_281.viewMatrix * light.lightPosition).xyz - _input.v.xyz;
    }
    float lightToSurfDist = length(L);
    L = normalize(L);
    float cosTheta = clamp(dot(N, L), 0.0, 1.0);
    float visibility = 1.0;
    float lightToSurfAngle = acos(dot(L, -light_dir));
    float param = lightToSurfAngle;
    int param_1 = int(light.lightAngleAttenCurveType);
    float4 param_2[2];
    spvArrayCopy(param_2, light.lightAngleAttenCurveParams);
    float atten = apply_atten_curve(param, param_1, param_2);
    float param_3 = lightToSurfDist;
    int param_4 = int(light.lightDistAttenCurveType);
    float4 param_5[2];
    spvArrayCopy(param_5, light.lightDistAttenCurveParams);
    atten *= apply_atten_curve(param_3, param_4, param_5);
    float3 R = normalize((N * (2.0 * dot(L, N))) - L);
    float3 V = normalize(-_input.v.xyz);
    float3 admit_light = light.lightColor.xyz * (light.lightIntensity * atten);
    float3 linearColor = diffuseMap.sample(samp0, _input.uv).xyz * cosTheta;
    if (visibility > 0.20000000298023223876953125)
    {
        linearColor += float3(0.800000011920928955078125 * pow(clamp(dot(R, V), 0.0, 1.0), 50.0));
    }
    linearColor *= admit_light;
    return linearColor * visibility;
}

float3 exposure_tone_mapping(thread const float3& color)
{
    return float3(1.0) - exp((-color) * 1.0);
}

float4 _basic_frag_main(thread const vert_output& _input, constant PerFrameConstants& v_281, thread texture2d<float> diffuseMap, thread sampler samp0, constant LightInfo& v_682, thread texturecube_array<float> skybox)
{
    float3 linearColor = float3(0.0);
    for (uint i = 0u; i < v_281.numLights; i++)
    {
        if (v_682.lights[i].lightType == 3u)
        {
            Light arg;
            arg.lightIntensity = v_682.lights[i].lightIntensity;
            arg.lightType = v_682.lights[i].lightType;
            arg.lightCastShadow = v_682.lights[i].lightCastShadow;
            arg.lightShadowMapIndex = v_682.lights[i].lightShadowMapIndex;
            arg.lightAngleAttenCurveType = v_682.lights[i].lightAngleAttenCurveType;
            arg.lightDistAttenCurveType = v_682.lights[i].lightDistAttenCurveType;
            arg.lightSize = v_682.lights[i].lightSize;
            arg.lightGuid = v_682.lights[i].lightGuid;
            arg.lightPosition = v_682.lights[i].lightPosition;
            arg.lightColor = v_682.lights[i].lightColor;
            arg.lightDirection = v_682.lights[i].lightDirection;
            arg.lightDistAttenCurveParams[0] = v_682.lights[i].lightDistAttenCurveParams[0];
            arg.lightDistAttenCurveParams[1] = v_682.lights[i].lightDistAttenCurveParams[1];
            arg.lightAngleAttenCurveParams[0] = v_682.lights[i].lightAngleAttenCurveParams[0];
            arg.lightAngleAttenCurveParams[1] = v_682.lights[i].lightAngleAttenCurveParams[1];
            arg.lightVP = v_682.lights[i].lightVP;
            arg.padding[0] = v_682.lights[i].padding[0];
            arg.padding[1] = v_682.lights[i].padding[1];
            vert_output param = _input;
            linearColor += apply_areaLight(arg, param, v_281, diffuseMap, samp0);
        }
        else
        {
            Light arg_1;
            arg_1.lightIntensity = v_682.lights[i].lightIntensity;
            arg_1.lightType = v_682.lights[i].lightType;
            arg_1.lightCastShadow = v_682.lights[i].lightCastShadow;
            arg_1.lightShadowMapIndex = v_682.lights[i].lightShadowMapIndex;
            arg_1.lightAngleAttenCurveType = v_682.lights[i].lightAngleAttenCurveType;
            arg_1.lightDistAttenCurveType = v_682.lights[i].lightDistAttenCurveType;
            arg_1.lightSize = v_682.lights[i].lightSize;
            arg_1.lightGuid = v_682.lights[i].lightGuid;
            arg_1.lightPosition = v_682.lights[i].lightPosition;
            arg_1.lightColor = v_682.lights[i].lightColor;
            arg_1.lightDirection = v_682.lights[i].lightDirection;
            arg_1.lightDistAttenCurveParams[0] = v_682.lights[i].lightDistAttenCurveParams[0];
            arg_1.lightDistAttenCurveParams[1] = v_682.lights[i].lightDistAttenCurveParams[1];
            arg_1.lightAngleAttenCurveParams[0] = v_682.lights[i].lightAngleAttenCurveParams[0];
            arg_1.lightAngleAttenCurveParams[1] = v_682.lights[i].lightAngleAttenCurveParams[1];
            arg_1.lightVP = v_682.lights[i].lightVP;
            arg_1.padding[0] = v_682.lights[i].padding[0];
            arg_1.padding[1] = v_682.lights[i].padding[1];
            vert_output param_1 = _input;
            linearColor += apply_light(arg_1, param_1, v_281, diffuseMap, samp0);
        }
    }
    linearColor += (skybox.sample(samp0, float4(_input.normal_world.xyz, 0.0).xyz, uint(round(float4(_input.normal_world.xyz, 0.0).w)), level(2.0)).xyz * float3(0.20000000298023223876953125));
    float3 param_2 = linearColor;
    linearColor = exposure_tone_mapping(param_2);
    return float4(linearColor, 1.0);
}

fragment basic_frag_main_out basic_frag_main(basic_frag_main_in in [[stage_in]], constant PerFrameConstants& v_281 [[buffer(10)]], constant LightInfo& v_682 [[buffer(12)]], texture2d<float> diffuseMap [[texture(0)]], texturecube_array<float> skybox [[texture(10)]], sampler samp0 [[sampler(0)]], float4 gl_FragCoord [[position]])
{
    basic_frag_main_out out = {};
    float3x3 input_TBN = {};
    input_TBN[0] = in.input_TBN_0;
    input_TBN[1] = in.input_TBN_1;
    input_TBN[2] = in.input_TBN_2;
    vert_output _input;
    _input.pos = gl_FragCoord;
    _input.normal = in.input_normal;
    _input.normal_world = in.input_normal_world;
    _input.v = in.input_v;
    _input.v_world = in.input_v_world;
    _input.uv = in.input_uv;
    _input.TBN = input_TBN;
    _input.v_tangent = in.input_v_tangent;
    _input.camPos_tangent = in.input_camPos_tangent;
    vert_output param = _input;
    out._entryPointOutput = _basic_frag_main(param, v_281, diffuseMap, samp0, v_682, skybox);
    return out;
}

