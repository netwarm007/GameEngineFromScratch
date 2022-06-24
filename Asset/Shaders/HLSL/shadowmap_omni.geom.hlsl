#include "cbuffer.h"
#include "vsoutput.h.hlsl"

struct gs_layered_output
{
    float4 pos : SV_POSITION;
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
        output.slice = uint(shadowmap_layer_index) * 6 + face; // built-in variable that specifies to which face we render.
        for(int i = 0; i < 3; ++i) // for each triangle's vertices
        {
            output.pos = mul(_entryPointOutput[i].pos, shadowMatrices[face]);
            OutputStream.Append(output);
        }    
        OutputStream.RestartStrip();
    }
}
