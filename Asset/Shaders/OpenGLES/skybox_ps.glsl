#version 310 es
precision mediump float;
precision highp int;

struct Light
{
    int lightType;
    highp float lightIntensity;
    uint lightCastShadow;
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
} _89;

layout(binding = 1, std140) uniform PerBatchConstants
{
    highp mat4 modelMatrix;
} _92;

layout(binding = 4) uniform highp samplerCubeArray skybox;
layout(binding = 0) uniform highp sampler2D diffuseMap;
layout(binding = 1) uniform highp sampler2DArray shadowMap;
layout(binding = 2) uniform highp sampler2DArray globalShadowMap;
layout(binding = 3) uniform highp samplerCubeArray cubeShadowMap;
layout(binding = 5) uniform highp sampler2D normalMap;
layout(binding = 6) uniform highp sampler2D metallicMap;
layout(binding = 7) uniform highp sampler2D roughnessMap;
layout(binding = 8) uniform highp sampler2D aoMap;
layout(binding = 9) uniform highp sampler2D brdfLUT;

layout(location = 0) out highp vec4 outputColor;
layout(location = 0) in highp vec3 UVW;

highp vec3 inverse_gamma_correction(highp vec3 color)
{
    return pow(color, vec3(2.2000000476837158203125));
}

highp vec3 exposure_tone_mapping(highp vec3 color)
{
    return vec3(1.0) - exp((-color) * 1.0);
}

highp vec3 gamma_correction(highp vec3 color)
{
    return pow(color, vec3(0.4545454680919647216796875));
}

void main()
{
    outputColor = textureLod(skybox, vec4(UVW, 0.0), 0.0);
    highp vec3 param = outputColor.xyz;
    highp vec3 _60 = inverse_gamma_correction(param);
    outputColor = vec4(_60.x, _60.y, _60.z, outputColor.w);
    highp vec3 param_1 = outputColor.xyz;
    highp vec3 _66 = exposure_tone_mapping(param_1);
    outputColor = vec4(_66.x, _66.y, _66.z, outputColor.w);
    highp vec3 param_2 = outputColor.xyz;
    highp vec3 _72 = gamma_correction(param_2);
    outputColor = vec4(_72.x, _72.y, _72.z, outputColor.w);
}

