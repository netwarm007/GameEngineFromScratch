#version 310 es

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

layout(binding = 1, std140) uniform PerBatchConstants
{
    mat4 modelMatrix;
} _13;

layout(binding = 0, std140) uniform PerFrameConstants
{
    mat4 viewMatrix;
    mat4 projectionMatrix;
    vec4 camPos;
    int numLights;
    Light allLights[100];
} _42;

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

layout(location = 3) out vec4 v_world;
layout(location = 0) in vec3 inputPosition;
layout(location = 2) out vec4 v;
layout(location = 1) out vec4 normal_world;
layout(location = 1) in vec3 inputNormal;
layout(location = 0) out vec4 normal;
layout(location = 4) out vec2 uv;
layout(location = 2) in vec2 inputUV;

void main()
{
    v_world = _13.modelMatrix * vec4(inputPosition, 1.0);
    v = _42.viewMatrix * v_world;
    gl_Position = _42.projectionMatrix * v;
    normal_world = _13.modelMatrix * vec4(inputNormal, 0.0);
    normal = _42.viewMatrix * normal_world;
    uv.x = inputUV.x;
    uv.y = 1.0 - inputUV.y;
}

