#version 310 es
precision mediump float;
precision highp int;

struct Light
{
    highp float lightIntensity;
    int lightType;
    int lightCastShadow;
    int lightShadowMapIndex;
    int lightAngleAttenCurveType;
    int lightDistAttenCurveType;
    highp vec2 lightSize;
    ivec4 lightGUID;
    highp vec4 lightPosition;
    highp vec4 lightColor;
    highp vec4 lightDirection;
    highp vec4 lightDistAttenCurveParams[2];
    highp vec4 lightAngleAttenCurveParams[2];
    highp mat4 lightVP;
    highp vec4 padding[2];
};

layout(binding = 0, std140) uniform PerFrameConstants
{
    highp mat4 viewMatrix;
    highp mat4 projectionMatrix;
    highp vec4 camPos;
    int numLights;
    Light allLights[100];
} _193;

layout(location = 3) in highp vec4 v_world;
layout(location = 1) in highp vec4 normal_world;
layout(location = 0) out highp vec4 outputColor;

float _52;

highp float linear_interpolate(highp float t, highp float begin, highp float end)
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

highp float apply_atten_curve(highp float dist, int atten_curve_type, highp vec4 atten_params[2])
{
    highp float atten = 1.0;
    switch (atten_curve_type)
    {
        case 1:
        {
            highp float begin_atten = atten_params[0].x;
            highp float end_atten = atten_params[0].y;
            highp float param = dist;
            highp float param_1 = begin_atten;
            highp float param_2 = end_atten;
            atten = linear_interpolate(param, param_1, param_2);
            break;
        }
        case 2:
        {
            highp float begin_atten_1 = atten_params[0].x;
            highp float end_atten_1 = atten_params[0].y;
            highp float param_3 = dist;
            highp float param_4 = begin_atten_1;
            highp float param_5 = end_atten_1;
            highp float tmp = linear_interpolate(param_3, param_4, param_5);
            atten = (3.0 * pow(tmp, 2.0)) - (2.0 * pow(tmp, 3.0));
            break;
        }
        case 3:
        {
            highp float scale = atten_params[0].x;
            highp float offset = atten_params[0].y;
            highp float kl = atten_params[0].z;
            highp float kc = atten_params[0].w;
            atten = clamp((scale / ((kl * dist) + (kc * scale))) + offset, 0.0, 1.0);
            break;
        }
        case 4:
        {
            highp float scale_1 = atten_params[0].x;
            highp float offset_1 = atten_params[0].y;
            highp float kq = atten_params[0].z;
            highp float kl_1 = atten_params[0].w;
            highp float kc_1 = atten_params[1].x;
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

void main()
{
    highp vec3 Lo = vec3(0.0);
    for (int i = 0; i < _193.numLights; i++)
    {
        Light light;
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
        highp vec3 L = normalize(light.lightPosition.xyz - v_world.xyz);
        highp vec3 N = normal_world.xyz;
        highp float NdotL = max(dot(N, L), 0.0);
        highp float visibility = 1.0;
        highp float lightToSurfDist = length(L);
        highp float lightToSurfAngle = acos(dot(-L, light.lightDirection.xyz));
        highp float param = lightToSurfAngle;
        int param_1 = light.lightAngleAttenCurveType;
        highp vec4 param_2[2] = light.lightAngleAttenCurveParams;
        highp float atten = apply_atten_curve(param, param_1, param_2);
        highp float param_3 = lightToSurfDist;
        int param_4 = light.lightDistAttenCurveType;
        highp vec4 param_5[2] = light.lightDistAttenCurveParams;
        atten *= apply_atten_curve(param_3, param_4, param_5);
        highp vec3 radiance = light.lightColor.xyz * (light.lightIntensity * atten);
        Lo += ((radiance * NdotL) * visibility);
    }
    outputColor = vec4(Lo, 1.0);
}

