#include "cbuffer.h"
#include "vsoutput.h.hlsl"

cube_vert_output passthrough_cube_vert_main(a2v_cube a)
{
    cube_vert_output o;
    o.pos = float4(a.inputPosition, 1.0f);
    o.uvw = a.inputUVW;

    return o;
}
