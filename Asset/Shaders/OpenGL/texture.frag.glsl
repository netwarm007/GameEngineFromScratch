#version 420

struct simple_vert_output
{
    vec4 pos;
    vec2 uv;
};

uniform sampler2D SPIRV_Cross_Combinedtexsamp0;

layout(location = 0) in vec2 input_uv;
layout(location = 0) out vec4 _entryPointOutput;

vec4 _texture_frag_main(simple_vert_output _input)
{
    return texture(SPIRV_Cross_Combinedtexsamp0, _input.uv);
}

void main()
{
    simple_vert_output _input;
    _input.pos = gl_FragCoord;
    _input.uv = input_uv;
    simple_vert_output param = _input;
    _entryPointOutput = _texture_frag_main(param);
}

