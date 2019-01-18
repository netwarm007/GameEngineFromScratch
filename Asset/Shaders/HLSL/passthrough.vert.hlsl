#include "cbuffer.h"
#include "vsoutput.h.hlsl"

simple_vert_output passthrough_vert_main(a2v_simple a)
{
    simple_vert_output o;
    o.pos = float4(a.inputPosition, 1.0f);
    o.uv = a.inputUV;

    return o;
}
