#version 400
#ifdef GL_ARB_shading_language_420pack
#extension GL_ARB_shading_language_420pack : require
#endif

struct Light
{
    int lightType;
    vec4 lightPosition;
    vec4 lightColor;
    vec4 lightDirection;
    vec4 lightSize;
    float lightIntensity;
    mat4 lightDistAttenCurveParams;
    mat4 lightAngleAttenCurveParams;
    mat4 lightVP;
    int lightShadowMapIndex;
};

layout(binding = 1, std140) uniform DrawBatchConstants
{
    mat4 modelMatrix;
    vec3 diffuseColor;
    vec3 specularColor;
    float specularPower;
    float metallic;
    float roughness;
    float ao;
    uint usingDiffuseMap;
    uint usingNormalMap;
    uint usingMetallicMap;
    uint usingRoughnessMap;
    uint usingAoMap;
} _15;

layout(binding = 0, std140) uniform DrawFrameConstants
{
    mat4 viewMatrix;
    mat4 projectionMatrix;
    vec3 ambientColor;
    vec3 camPos;
    int numLights;
    Light allLights[100];
} _36;

layout(binding = 0) uniform sampler2D diffuseMap;
layout(binding = 1) uniform sampler2DArray shadowMap;
layout(binding = 2) uniform sampler2DArray globalShadowMap;
layout(binding = 3) uniform samplerCubeArray cubeShadowMap;
layout(binding = 4) uniform samplerCubeArray skybox;
layout(binding = 5) uniform sampler2D normalMap;
layout(binding = 6) uniform sampler2D metallicMap;
layout(binding = 7) uniform sampler2D roughnessMap;
layout(binding = 8) uniform sampler2D aoMap;
layout(binding = 9) uniform sampler2D brdfLUT;

out vec4 v_world;
layout(location = 0) in vec3 inputPosition;
out vec4 v;
out vec4 normal_world;
layout(location = 1) in vec3 inputNormal;
out vec4 normal;
out vec2 uv;
layout(location = 2) in vec2 inputUV;

void main()
{
    v_world = _15.modelMatrix * vec4(inputPosition, 1.0);
    v = _36.viewMatrix * v_world;
    gl_Position = _36.projectionMatrix * v;
    normal_world = _15.modelMatrix * vec4(inputNormal, 0.0);
    normal = _36.viewMatrix * normal_world;
    uv.x = inputUV.x;
    uv.y = 1.0 - inputUV.y;
}

