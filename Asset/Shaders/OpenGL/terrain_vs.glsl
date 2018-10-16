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

layout(std140) uniform PerFrameConstants
{
    mat4 viewMatrix;
    mat4 projectionMatrix;
    vec4 camPos;
    int numLights;
    Light allLights[100];
} _34;

out vec4 v_world;
layout(location = 0) in vec3 inputPosition;
out vec4 v;
out vec4 normal_world;
out vec4 normal;
out mat3 TBN;
out vec3 v_tangent;
out vec3 camPos_tangent;
out vec2 uv;
layout(location = 1) in vec2 inputUV;

void main()
{
    v_world = vec4(inputPosition, 1.0);
    v = _34.viewMatrix * v_world;
    gl_Position = _34.projectionMatrix * v;
    normal_world = vec4(0.0, 0.0, 1.0, 0.0);
    normal = normalize(_34.viewMatrix * normal_world);
    vec3 tangent = vec3(1.0, 0.0, 0.0);
    vec3 bitangent = vec3(0.0, 1.0, 0.0);
    tangent = normalize(tangent - (normal_world.xyz * dot(tangent, normal_world.xyz)));
    bitangent = cross(normal_world.xyz, tangent);
    TBN = mat3(vec3(tangent), vec3(bitangent), vec3(normal_world.xyz));
    mat3 TBN_trans = transpose(TBN);
    v_tangent = TBN_trans * v_world.xyz;
    camPos_tangent = TBN_trans * _34.camPos.xyz;
    uv.x = inputUV.x;
    uv.y = 1.0 - inputUV.y;
}

