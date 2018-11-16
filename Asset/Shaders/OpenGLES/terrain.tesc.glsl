#version 310 es
#extension GL_EXT_tessellation_shader : require
layout(vertices = 4) out;

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

layout(binding = 1, std140) uniform PerFrameConstants
{
    mat4 viewMatrix;
    mat4 projectionMatrix;
    vec4 camPos;
    int numLights;
    Light allLights[100];
} _43;

vec4 project(vec4 vertex)
{
    vec4 result = (_43.projectionMatrix * _43.viewMatrix) * vertex;
    result /= vec4(result.w);
    return result;
}

bool offscreen(vec4 vertex)
{
    if (vertex.z < (-0.5))
    {
        return true;
    }
    bool _94 = any(lessThan(vertex.xy, vec2(-1.7000000476837158203125)));
    bool _104;
    if (!_94)
    {
        _104 = any(greaterThan(vertex.xy, vec2(1.7000000476837158203125)));
    }
    else
    {
        _104 = _94;
    }
    return _104;
}

vec2 screen_space(vec4 vertex)
{
    return (clamp(vertex.xy, vec2(-1.2999999523162841796875), vec2(1.2999999523162841796875)) + vec2(1.0)) * vec2(480.0, 270.0);
}

float level(vec2 v0, vec2 v1)
{
    return clamp(distance(v0, v1) / 2.0, 1.0, 64.0);
}

void main()
{
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
    if (gl_InvocationID == 0)
    {
        vec4 param = gl_in[0].gl_Position;
        vec4 v0 = project(param);
        vec4 param_1 = gl_in[1].gl_Position;
        vec4 v1 = project(param_1);
        vec4 param_2 = gl_in[2].gl_Position;
        vec4 v2 = project(param_2);
        vec4 param_3 = gl_in[3].gl_Position;
        vec4 v3 = project(param_3);
        vec4 param_4 = v0;
        vec4 param_5 = v1;
        vec4 param_6 = v2;
        vec4 param_7 = v3;
        if (all(bvec4(offscreen(param_4), offscreen(param_5), offscreen(param_6), offscreen(param_7))))
        {
            gl_TessLevelInner[0] = 0.0;
            gl_TessLevelInner[1] = 0.0;
            gl_TessLevelOuter[0] = 0.0;
            gl_TessLevelOuter[1] = 0.0;
            gl_TessLevelOuter[2] = 0.0;
            gl_TessLevelOuter[3] = 0.0;
        }
        else
        {
            vec4 param_8 = v0;
            vec2 ss0 = screen_space(param_8);
            vec4 param_9 = v1;
            vec2 ss1 = screen_space(param_9);
            vec4 param_10 = v2;
            vec2 ss2 = screen_space(param_10);
            vec4 param_11 = v3;
            vec2 ss3 = screen_space(param_11);
            vec2 param_12 = ss1;
            vec2 param_13 = ss2;
            float e0 = level(param_12, param_13);
            vec2 param_14 = ss0;
            vec2 param_15 = ss1;
            float e1 = level(param_14, param_15);
            vec2 param_16 = ss3;
            vec2 param_17 = ss0;
            float e2 = level(param_16, param_17);
            vec2 param_18 = ss2;
            vec2 param_19 = ss3;
            float e3 = level(param_18, param_19);
            gl_TessLevelInner[0] = mix(e1, e2, 0.5);
            gl_TessLevelInner[1] = mix(e0, e3, 0.5);
            gl_TessLevelOuter[0] = e0;
            gl_TessLevelOuter[1] = e1;
            gl_TessLevelOuter[2] = e2;
            gl_TessLevelOuter[3] = e3;
        }
    }
}

