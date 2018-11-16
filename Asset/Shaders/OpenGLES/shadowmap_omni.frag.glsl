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

struct ps_constant_t
{
    highp vec3 lightPos;
    highp float far_plane;
};

uniform ps_constant_t u_lightParams;

layout(location = 0) in highp vec4 FragPos;

void main()
{
    highp float lightDistance = length(FragPos.xyz - u_lightParams.lightPos);
    lightDistance /= u_lightParams.far_plane;
    gl_FragDepth = lightDistance;
}

