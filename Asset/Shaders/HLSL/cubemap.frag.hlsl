#include "cbuffer.h"
#include "vsoutput.h.hlsl"

TextureCube cubemap : register(t0);

[RootSignature(MyRS1)]
float4 cubemap_frag_main(cube_vert_output _entryPointOutput) : SV_Target
{
    return cubemap.SampleLevel(samp0, _entryPointOutput.uvw, mip_level);
}
