#version 400

struct Light
{
    int lightType;
    float lightIntensity;
    int lightCastShadow;
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

struct ps_constant_t
{
    vec3 lightPos;
    float far_plane;
};

uniform ps_constant_t u_lightParams;

in vec4 FragPos;

void main()
{
    float lightDistance = length(FragPos.xyz - u_lightParams.lightPos);
    lightDistance /= u_lightParams.far_plane;
    gl_FragDepth = lightDistance;
}

