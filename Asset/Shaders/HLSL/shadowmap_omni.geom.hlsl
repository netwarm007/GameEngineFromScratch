#include "cbuffer.h"
#include "vsoutput.h.hlsl"

struct gs_layered_output
{
    float4 pos : SV_POSITION;
#if defined(OS_WINDOWS) && !defined(MYGE_GLSLANG)
    float3 worldPos : TEXCOORD0;
#endif
    uint slice  : SV_RENDERTARGETARRAYINDEX;
};

////////////////////////////////////////////////////////////////////////////////
// Geometry Shader
////////////////////////////////////////////////////////////////////////////////
[maxvertexcount(18)]
void shadowmap_omni_geom_main(triangle pos_only_vert_output _entryPointOutput[3], inout TriangleStream<gs_layered_output> OutputStream)
{
    gs_layered_output output;

    for(int face = 0; face < 6; face++)
    {
#if defined(OS_WINDOWS) && !defined(MYGE_GLSLANG)
        output.slice = face;
#else
        output.slice = uint(shadowmap_layer_index) * 6 + face;
#endif
        for(int i = 0; i < 3; ++i) // for each triangle's vertices
        {
            output.pos = mul(_entryPointOutput[i].pos, shadowMatrices[face]);
#if defined(OS_WINDOWS) && !defined(MYGE_GLSLANG)
            output.worldPos = _entryPointOutput[i].pos.xyz;
#endif
            OutputStream.Append(output);
        }    
        OutputStream.RestartStrip();
    }
}
