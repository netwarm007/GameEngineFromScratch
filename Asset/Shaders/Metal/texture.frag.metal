#pragma clang diagnostic ignored "-Wmissing-prototypes"

#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct simple_vert_output
{
    float4 pos;
    float2 uv;
};

struct texture_frag_main_out
{
    float4 _entryPointOutput [[color(0)]];
};

struct texture_frag_main_in
{
    float2 input_uv [[user(locn0)]];
};

float4 _texture_frag_main(thread const simple_vert_output& _input, thread texture2d<float> tex, thread sampler samp0)
{
    return tex.sample(samp0, _input.uv);
}

fragment texture_frag_main_out texture_frag_main(texture_frag_main_in in [[stage_in]], texture2d<float> tex [[texture(0)]], sampler samp0 [[sampler(0)]], float4 gl_FragCoord [[position]])
{
    texture_frag_main_out out = {};
    simple_vert_output _input;
    _input.pos = gl_FragCoord;
    _input.uv = in.input_uv;
    simple_vert_output param = _input;
    out._entryPointOutput = _texture_frag_main(param, tex, samp0);
    return out;
}

