#include "cbuffer.h"
#include "vsoutput.h.hlsl"

float4 shadowmap_omni_frag_main() : SV_TARGET
{
    // get distance between fragment and light source
    float lightDistance = length(FragPos.xyz - u_lightParams.lightPos);
    
    // map to [0;1] range by dividing by far_plane
    lightDistance = lightDistance / u_lightParams.far_plane;
    
    // write this as modified depth
    gl_FragDepth = lightDistance;
}  