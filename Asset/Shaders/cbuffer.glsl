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
    vec2 lightSize;
    float lightIntensity;
    int  lightDistAttenCurveType;
    float lightDistAttenCurveParams[5];
    int  lightAngleAttenCurveType;
    float lightAngleAttenCurveParams[5];
    int  lightShadowMapIndex;
    mat4 lightVP;
};

uniform DrawFrameConstants {
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
