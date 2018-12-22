#include "vsoutput.h.hlsl"

SamplerState samp0 : s0;
Texture2D tex : t0;

float4 texture_frag_main(simple_vert_output input) : SV_Target
{
    return tex.Sample(samp0, input.uv);
}
