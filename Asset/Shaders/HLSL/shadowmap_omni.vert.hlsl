#include "cbuffer.h"
#include "vsoutput.h.hlsl"

////////////////////////////////////////////////////////////////////////////////
// Vertex Shader
////////////////////////////////////////////////////////////////////////////////
#if defined(MYGE_POINT_SHADOW_SINGLE_FACE)
struct point_shadow_single_output
{
    float4 pos : SV_POSITION;
    float3 worldPos : TEXCOORD0;
};

point_shadow_single_output shadowmap_omni_single_vert_main(a2v_pos_only a)
{
    point_shadow_single_output o;
    float4 world = mul(float4(a.inputPosition, 1.0f), modelMatrix);
    o.pos = mul(world, shadowMatrices[0]);
    o.worldPos = world.xyz;
    return o;
}
#else
pos_only_vert_output shadowmap_omni_vert_main(a2v_pos_only a)
{
    pos_only_vert_output o;
	// Calculate the position of the vertex against the world, view, and projection matrices.
	float4 v = float4(a.inputPosition, 1.0f);
	o.pos = mul(v, modelMatrix);

    return o;
}
#endif
