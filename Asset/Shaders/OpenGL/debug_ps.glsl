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

layout(std140) uniform PerFrameConstants
{
    mat4 viewMatrix;
    mat4 projectionMatrix;
    vec4 camPos;
    int numLights;
    Light allLights[100];
} _37;

layout(std140) uniform PerBatchConstants
{
    mat4 modelMatrix;
} _40;

struct debugPushConstants
{
    vec3 FrontColor;
};

uniform debugPushConstants u_pushConstants;

uniform sampler2D diffuseMap;
uniform sampler2DArray shadowMap;
uniform sampler2DArray globalShadowMap;
uniform samplerCubeArray cubeShadowMap;
uniform samplerCubeArray skybox;
uniform sampler2D normalMap;
uniform sampler2D metallicMap;
uniform sampler2D roughnessMap;
uniform sampler2D aoMap;
uniform sampler2D brdfLUT;

layout(location = 0) out vec4 outputColor;

void main()
{
    outputColor = vec4(u_pushConstants.FrontColor, 1.0);
}

