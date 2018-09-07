#version 400 core

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
    mat4 lightDistAttenCurveParams;
    mat4 lightAngleAttenCurveParams;
    mat4 lightVP;
    int  lightShadowMapIndex;
};

uniform DrawFrameConstants {
    mat4 viewMatrix;
    mat4 projectionMatrix;
    vec3 ambientColor;
    int numLights;
    Light allLights[MAX_LIGHTS];
};

// samplers
uniform sampler2D diffuseMap;
uniform sampler2DArray shadowMap;
uniform sampler2DArray globalShadowMap;
uniform samplerCubeArray cubeShadowMap;

// per drawcall
uniform mat4 modelMatrix;

uniform vec3 diffuseColor;
uniform vec3 specularColor;
uniform float specularPower;

uniform bool usingDiffuseMap;

