#version 320 es

struct a2v_pos_only
{
    vec3 inputPosition;
};

struct pos_only_vert_output
{
    vec4 pos;
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

layout(binding = 10, std140) uniform PerFrameConstants
{
    mat4 viewMatrix;
    mat4 projectionMatrix;
    vec4 camPos;
    int numLights;
} _31;

layout(location = 0) in vec3 a_inputPosition;

pos_only_vert_output _debug_vert_main(a2v_pos_only a)
{
    vec4 v = vec4(a.inputPosition, 1.0);
    v = _31.viewMatrix * v;
    pos_only_vert_output o;
    o.pos = _31.projectionMatrix * v;
    return o;
}

void main()
{
    a2v_pos_only a;
    a.inputPosition = a_inputPosition;
    a2v_pos_only param = a;
    gl_Position = _debug_vert_main(param).pos;
}

