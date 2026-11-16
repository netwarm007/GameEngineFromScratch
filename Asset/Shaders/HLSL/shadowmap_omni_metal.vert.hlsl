#include "cbuffer.h"

struct point_shadow_output
{
    float4 pos : SV_POSITION;
    float3 worldPos : TEXCOORD0;
    uint slice : SV_RENDERTARGETARRAYINDEX;
};

point_shadow_output shadowmap_omni_metal_vert_main(a2v_pos_only a, uint face : SV_InstanceID)
{
    point_shadow_output o;
    float4 world = mul(float4(a.inputPosition, 1.0f), modelMatrix);
    o.pos = mul(world, shadowMatrices[face]);
    o.worldPos = world.xyz;
    o.slice = face;
    return o;
}
