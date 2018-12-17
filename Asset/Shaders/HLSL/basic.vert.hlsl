#include "cbuffer.h"
#include "vsoutput.h.hlsl"

basic_vert_output basic_vert_main(a2v a)
{
    basic_vert_output o;

    o.v_world = mul(float4(a.inputPosition, 1.0f), modelMatrix);
    o.v = mul(o.v_world, viewMatrix);
    o.pos = mul(o.v, projectionMatrix);
    o.normal_world = normalize(mul(float4(a.inputNormal, 0.0f), modelMatrix));
    o.normal = normalize(mul(o.normal_world, viewMatrix));
    o.uv.x = a.inputUV.x;
    o.uv.y = 1.0f - a.inputUV.y;

    return o;
}
