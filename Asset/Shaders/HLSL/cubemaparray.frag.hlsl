#include "cbuffer.h"
#include "functions.h.hlsl"
#include "vsoutput.h.hlsl"

#if defined(OS_WEBASSEMBLY)
Texture2DArray cubemap : register(t0);
#else
TextureCubeArray cubemap : register(t0);
#endif

[RootSignature(MyRS1)]
float4 cubemaparray_frag_main(cube_vert_output _entryPointOutput) : SV_Target
{
#if defined(OS_WEBASSEMBLY)
    float3 uvw = convert_xyz_to_cube_uv(_entryPointOutput.uvw);
    uvw.z += layer_index * 6;
    return cubemap.SampleLevel(samp0, uvw, mip_level);
#else
    return cubemap.SampleLevel(samp0, float4(_entryPointOutput.uvw, layer_index), mip_level);
#endif
}
