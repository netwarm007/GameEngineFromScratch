#include "cbuffer.h"
#include "vsoutput.h.hlsl"

MYGE_ROOT_SIGNATURE(MyRS1)
#if defined(OS_WINDOWS) && !defined(MYGE_GLSLANG)
float shadowmap_omni_frag_main(float4 pos : SV_POSITION, float3 worldPos : TEXCOORD0) : SV_DEPTH
#else
float shadowmap_omni_frag_main(pos_only_vert_output _entryPointOutput) : SV_DEPTH
#endif
{
    // get distance between fragment and light source
#if defined(OS_WINDOWS) && !defined(MYGE_GLSLANG)
    float lightDistance = length(worldPos - lights[light_index].lightPosition.xyz);
#else
    float lightDistance = length(_entryPointOutput.pos.xyz - lights[light_index].lightPosition.xyz);
#endif
    
    // map to [0;1] range by dividing by far_plane
    lightDistance = lightDistance / far_plane;
    
    // write this as modified depth
    return lightDistance;
}  