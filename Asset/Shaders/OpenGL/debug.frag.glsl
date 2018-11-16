#version 400

struct vert_output
{
    vec4 position;
    vec4 normal;
    vec4 normal_world;
    vec4 v;
    vec4 v_world;
    vec2 uv;
    mat3 TBN;
    vec3 v_tangent;
    vec3 camPos_tangent;
};

struct Light
{
    float lightIntensity;
    uint lightType;
    int lightCastShadow;
    int lightShadowMapIndex;
    uint lightAngleAttenCurveType;
    uint lightDistAttenCurveType;
    vec2 lightSize;
    uvec4 lightGuid;
    vec4 lightPosition;
    vec4 lightColor;
    vec4 lightDirection;
    vec4 lightDistAttenCurveParams[2];
    vec4 lightAngleAttenCurveParams[2];
    mat4 lightVP;
    vec4 padding[2];
};

in vec4 input_normal;
in vec4 input_normal_world;
in vec4 input_v;
in vec4 input_v_world;
in vec2 input_uv;
in mat3 input_TBN;
in vec3 input_v_tangent;
in vec3 input_camPos_tangent;
layout(location = 0) out vec4 _entryPointOutput;

vec4 _debug_frag_main(vert_output _input)
{
    return vec4(1.0);
}

void main()
{
    vert_output _input;
    _input.position = gl_FragCoord;
    _input.normal = input_normal;
    _input.normal_world = input_normal_world;
    _input.v = input_v;
    _input.v_world = input_v_world;
    _input.uv = input_uv;
    _input.TBN = input_TBN;
    _input.v_tangent = input_v_tangent;
    _input.camPos_tangent = input_camPos_tangent;
    vert_output param = _input;
    _entryPointOutput = _debug_frag_main(param);
}

