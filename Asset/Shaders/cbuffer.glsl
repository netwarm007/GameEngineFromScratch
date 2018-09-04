#version 420 core

/////////////////////
// CONSTANTS       //
/////////////////////
// per frame
#define MAX_LIGHTS 100

struct Light {
    int  lightType;
    vec4 lightPosition;
    vec4 lightColor;
    vec4 lightDirection;
    vec4 lightSize;
    float lightIntensity;
    int  lightDistAttenCurveType;
    float lightDistAttenCurveParams_0;
    float lightDistAttenCurveParams_1;
    float lightDistAttenCurveParams_2;
    float lightDistAttenCurveParams_3;
    float lightDistAttenCurveParams_4;
    int  lightAngleAttenCurveType;
    float lightAngleAttenCurveParams_0;
    float lightAngleAttenCurveParams_1;
    float lightAngleAttenCurveParams_2;
    float lightAngleAttenCurveParams_3;
    float lightAngleAttenCurveParams_4;
    int  lightShadowMapIndex;
    mat4 lightVP;
};

layout(std140) uniform DrawFrameConstants {
    mat4 viewMatrix;
    mat4 projectionMatrix;
    vec3 ambientColor;
    vec3 camPos;
    int numLights;
    Light allLights[MAX_LIGHTS];
};

// samplers
layout(binding = 0) uniform sampler2D diffuseMap;
layout(binding = 1) uniform sampler2DArray shadowMap;
layout(binding = 2) uniform sampler2DArray globalShadowMap;
layout(binding = 3) uniform samplerCubeArray cubeShadowMap;
layout(binding = 4) uniform samplerCube skybox;

// per drawcall
uniform mat4 modelMatrix;

uniform vec3 diffuseColor;
uniform vec3 specularColor;
uniform float specularPower;
uniform float metallic;
uniform float roughness;
uniform float ao;

uniform bool usingDiffuseMap;

