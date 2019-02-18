#version 420
layout(triangles) in;
layout(max_vertices = 18, triangle_strip) out;

struct pos_only_vert_output
{
    vec4 pos;
};

struct gs_layered_output
{
    vec4 pos;
    int slice;
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

layout(binding = 14, std140) uniform ShadowMapConstants
{
    mat4 shadowMatrices[6];
    vec4 lightPos;
    float shadowmap_layer_index;
    float far_plane;
} _40;

void _shadowmap_omni_geom_main(pos_only_vert_output _entryPointOutput[3], gs_layered_output OutputStream)
{
    for (int face = 0; face < 6; face++)
    {
        gs_layered_output _output;
        _output.slice = (int(_40.shadowmap_layer_index) * 6) + face;
        for (int i = 0; i < 3; i++)
        {
            _output.pos = _40.shadowMatrices[face] * _entryPointOutput[i].pos;
            gl_Position = _output.pos;
            gl_Layer = _output.slice;
            EmitVertex();
        }
        EndPrimitive();
    }
}

void main()
{
    pos_only_vert_output _entryPointOutput[3];
    _entryPointOutput[0].pos = gl_in[0].gl_Position;
    _entryPointOutput[1].pos = gl_in[1].gl_Position;
    _entryPointOutput[2].pos = gl_in[2].gl_Position;
    pos_only_vert_output param[3] = _entryPointOutput;
    gs_layered_output param_1;
    _shadowmap_omni_geom_main(param, param_1);
    gs_layered_output OutputStream = param_1;
}

