#version 320 es

struct a2v
{
    vec3 inputPosition;
    vec3 inputNormal;
    vec2 inputUV;
    vec3 inputTangent;
};

struct pbr_vert_output
{
    vec4 pos;
    vec4 normal;
    vec4 normal_world;
    vec4 v;
    vec4 v_world;
    vec3 v_tangent;
    vec3 camPos_tangent;
    vec2 uv;
    mat3 TBN;
};

struct Light
{
    float lightIntensity;
    int lightType;
    int lightCastShadow;
    int lightShadowMapIndex;
    int lightAngleAttenCurveType;
    int lightDistAttenCurveType;
    vec2 lightSize;
    ivec4 lightGuid;
    vec4 lightPosition;
    vec4 lightColor;
    vec4 lightDirection;
    vec4 lightDistAttenCurveParams[2];
    vec4 lightAngleAttenCurveParams[2];
    mat4 lightVP;
    vec4 padding[2];
};

layout(binding = 11, std140) uniform PerBatchConstants
{
    mat4 modelMatrix;
} _25;

layout(binding = 10, std140) uniform PerFrameConstants
{
    mat4 viewMatrix;
    mat4 projectionMatrix;
    vec4 camPos;
    int numLights;
} _44;

layout(location = 0) in vec3 a_inputPosition;
layout(location = 1) in vec3 a_inputNormal;
layout(location = 2) in vec2 a_inputUV;
layout(location = 3) in vec3 a_inputTangent;
layout(location = 0) out vec4 _entryPointOutput_normal;
layout(location = 1) out vec4 _entryPointOutput_normal_world;
layout(location = 2) out vec4 _entryPointOutput_v;
layout(location = 3) out vec4 _entryPointOutput_v_world;
layout(location = 4) out vec3 _entryPointOutput_v_tangent;
layout(location = 5) out vec3 _entryPointOutput_camPos_tangent;
layout(location = 6) out vec2 _entryPointOutput_uv;
layout(location = 7) out mat3 _entryPointOutput_TBN;

pbr_vert_output _pbr_vert_main(a2v a)
{
    pbr_vert_output o;
    o.v_world = _25.modelMatrix * vec4(a.inputPosition, 1.0);
    o.v = _44.viewMatrix * o.v_world;
    o.pos = _44.projectionMatrix * o.v;
    o.normal_world = normalize(_25.modelMatrix * vec4(a.inputNormal, 0.0));
    o.normal = normalize(_44.viewMatrix * o.normal_world);
    vec3 tangent = normalize((_25.modelMatrix * vec4(a.inputTangent, 0.0)).xyz);
    tangent = normalize(tangent - (o.normal_world.xyz * dot(tangent, o.normal_world.xyz)));
    vec3 bitangent = cross(o.normal_world.xyz, tangent);
    o.TBN = mat3(vec3(tangent), vec3(bitangent), vec3(o.normal_world.xyz));
    mat3 TBN_trans = transpose(o.TBN);
    o.v_tangent = TBN_trans * o.v_world.xyz;
    o.camPos_tangent = TBN_trans * _44.camPos.xyz;
    o.uv.x = a.inputUV.x;
    o.uv.y = 1.0 - a.inputUV.y;
    return o;
}

void main()
{
    a2v a;
    a.inputPosition = a_inputPosition;
    a.inputNormal = a_inputNormal;
    a.inputUV = a_inputUV;
    a.inputTangent = a_inputTangent;
    a2v param = a;
    pbr_vert_output flattenTemp = _pbr_vert_main(param);
    gl_Position = flattenTemp.pos;
    _entryPointOutput_normal = flattenTemp.normal;
    _entryPointOutput_normal_world = flattenTemp.normal_world;
    _entryPointOutput_v = flattenTemp.v;
    _entryPointOutput_v_world = flattenTemp.v_world;
    _entryPointOutput_v_tangent = flattenTemp.v_tangent;
    _entryPointOutput_camPos_tangent = flattenTemp.camPos_tangent;
    _entryPointOutput_uv = flattenTemp.uv;
    _entryPointOutput_TBN = flattenTemp.TBN;
}

