#version 330 core

/////////////////////
// CONSTANTS       //
/////////////////////
// per frame
#define MAX_LIGHTS 100

struct Light {
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
    mat4 worldMatrix;
    mat4 viewMatrix;
    mat4 projectionMatrix;
    vec3 ambientColor;
    int numLights;
    Light allLights[MAX_LIGHTS];
};

uniform sampler2DArray shadowMap;

// per drawcall
uniform mat4 modelMatrix;

uniform vec3 diffuseColor;
uniform vec3 specularColor;
uniform float specularPower;

uniform bool usingDiffuseMap;

uniform sampler2D diffuseMap;
