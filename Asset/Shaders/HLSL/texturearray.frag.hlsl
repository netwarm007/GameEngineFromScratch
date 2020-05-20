#include "cbuffer.h"
#include "vsoutput.h.hlsl"

Texture2DArray texture_array : register(t0);

[RootSignature(MyRS1)]
float4 texturearray_frag_main(simple_vert_output _entryPointOutput) : SV_Target
{
    return texture_array.SampleLevel(samp0, float3(_entryPointOutput.uv, layer_index), mip_level);
}
