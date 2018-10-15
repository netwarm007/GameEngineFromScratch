#version 400

struct Light
{
    float lightIntensity;
    int lightType;
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

layout(std140) uniform PerBatchConstants
{
    mat4 modelMatrix;
} _13;

layout(std140) uniform PerFrameConstants
{
    mat4 viewMatrix;
    mat4 projectionMatrix;
    vec4 camPos;
    int numLights;
    Light allLights[100];
} _42;

out vec4 v_world;
layout(location = 0) in vec3 inputPosition;
out vec4 v;
out vec4 normal_world;
layout(location = 1) in vec3 inputNormal;
out vec4 normal;
layout(location = 3) in vec3 inputTangent;
out mat3 TBN;
out vec3 v_tangent;
out vec3 camPos_tangent;
out vec2 uv;
layout(location = 2) in vec2 inputUV;

void main()
{
    v_world = _13.modelMatrix * vec4(inputPosition, 1.0);
    v = _42.viewMatrix * v_world;
    gl_Position = _42.projectionMatrix * v;
    normal_world = normalize(_13.modelMatrix * vec4(inputNormal, 0.0));
    normal = normalize(_42.viewMatrix * normal_world);
    vec3 tangent = normalize(vec3((_13.modelMatrix * vec4(inputTangent, 0.0)).xyz));
    tangent = normalize(tangent - (normal_world.xyz * dot(tangent, normal_world.xyz)));
    vec3 bitangent = cross(normal_world.xyz, tangent);
    TBN = mat3(vec3(tangent), vec3(bitangent), vec3(normal_world.xyz));
    mat3 TBN_trans = transpose(TBN);
    v_tangent = TBN_trans * v_world.xyz;
    camPos_tangent = TBN_trans * _42.camPos.xyz;
    uv.x = inputUV.x;
    uv.y = 1.0 - inputUV.y;
}

