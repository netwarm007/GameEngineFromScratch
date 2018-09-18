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
} _47;

layout(binding = 1, std140) uniform PerBatchConstants
{
    highp mat4 modelMatrix;
} _50;

struct constants_t
{
    highp vec3 lightPos;
    highp float far_plane;
};

uniform constants_t u_pushConstants;

layout(binding = 0) uniform highp sampler2D diffuseMap;
layout(binding = 1) uniform highp sampler2DArray shadowMap;
layout(binding = 2) uniform highp sampler2DArray globalShadowMap;
layout(binding = 3) uniform highp samplerCubeArray cubeShadowMap;
layout(binding = 4) uniform highp samplerCubeArray skybox;
layout(binding = 5) uniform highp sampler2D normalMap;
layout(binding = 6) uniform highp sampler2D metallicMap;
layout(binding = 7) uniform highp sampler2D roughnessMap;
layout(binding = 8) uniform highp sampler2D aoMap;
layout(binding = 9) uniform highp sampler2D brdfLUT;

layout(location = 0) in highp vec4 FragPos;

void main()
{
    highp float lightDistance = length(FragPos.xyz - u_pushConstants.lightPos);
    lightDistance /= u_pushConstants.far_plane;
    gl_FragDepth = lightDistance;
}

