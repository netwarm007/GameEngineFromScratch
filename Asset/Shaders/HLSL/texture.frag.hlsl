#include "cbuffer.h"
#include "vsoutput.h.hlsl"

Texture2D tex : register(t0);

[RootSignature(MyRS1)]
float4 texture_frag_main(simple_vert_output _entryPointOutput) : SV_Target
{
    return tex.Sample(samp0, _entryPointOutput.uv);
}
