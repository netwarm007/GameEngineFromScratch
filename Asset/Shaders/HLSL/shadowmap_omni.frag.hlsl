#include "cbuffer.h"
#include "vsoutput.h.hlsl"

MYGE_ROOT_SIGNATURE(MyRS1)
float shadowmap_omni_frag_main(float4 pos : SV_POSITION, float3 worldPos : TEXCOORD0) : SV_DEPTH
{
    // get distance between fragment and light source
    float lightDistance = length(worldPos - lights[light_index].lightPosition.xyz);
    
    // map to [0;1] range by dividing by far_plane
    lightDistance = lightDistance / far_plane;
    
    // write this as modified depth
    return lightDistance;
}  