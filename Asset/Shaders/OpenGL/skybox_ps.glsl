#version 400
#ifdef GL_ARB_shading_language_420pack
#extension GL_ARB_shading_language_420pack : require
#endif

struct Light
{
    int lightType;
    float lightIntensity;
    uint lightCastShadow;
    int lightShadowMapIndex;
    int lightAngleAttenCurveType;
    int lightDistAttenCurveType;
    vec2 lightSize;
    ivec4 lightGUID;
    vec4 lightPosition;
    vec4 lightColor;
    vec4 lightDirection;
    vec4 lightDistAttenCurveParams[2];
    vec4 lightAngleAttenCurveParams[2];
    mat4 lightVP;
    vec4 padding[2];
};

layout(binding = 0, std140) uniform PerFrameConstants
{
    mat4 viewMatrix;
    mat4 projectionMatrix;
    vec4 camPos;
    int numLights;
    Light allLights[100];
} _89;

layout(binding = 1, std140) uniform PerBatchConstants
{
    mat4 modelMatrix;
} _92;

layout(binding = 4) uniform samplerCubeArray skybox;
layout(binding = 0) uniform sampler2D diffuseMap;
layout(binding = 1) uniform sampler2DArray shadowMap;
layout(binding = 2) uniform sampler2DArray globalShadowMap;
layout(binding = 3) uniform samplerCubeArray cubeShadowMap;
layout(binding = 5) uniform sampler2D normalMap;
layout(binding = 6) uniform sampler2D metallicMap;
layout(binding = 7) uniform sampler2D roughnessMap;
layout(binding = 8) uniform sampler2D aoMap;
layout(binding = 9) uniform sampler2D brdfLUT;

layout(location = 0) out vec4 outputColor;
in vec3 UVW;

vec3 inverse_gamma_correction(vec3 color)
{
    return pow(color, vec3(2.2000000476837158203125));
}

vec3 exposure_tone_mapping(vec3 color)
{
    return vec3(1.0) - exp((-color) * 1.0);
}

vec3 gamma_correction(vec3 color)
{
    return pow(color, vec3(0.4545454680919647216796875));
}

void main()
{
    outputColor = textureLod(skybox, vec4(UVW, 0.0), 0.0);
    vec3 param = outputColor.xyz;
    vec3 _60 = inverse_gamma_correction(param);
    outputColor = vec4(_60.x, _60.y, _60.z, outputColor.w);
    vec3 param_1 = outputColor.xyz;
    vec3 _66 = exposure_tone_mapping(param_1);
    outputColor = vec4(_66.x, _66.y, _66.z, outputColor.w);
    vec3 param_2 = outputColor.xyz;
    vec3 _72 = gamma_correction(param_2);
    outputColor = vec4(_72.x, _72.y, _72.z, outputColor.w);
}

