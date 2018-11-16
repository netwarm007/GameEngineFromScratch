#include "cbuffer.h"
#include "vsoutput.h.hlsl"

////////////////////////////////////////////////////////////////////////////////
// Vertex Shader
////////////////////////////////////////////////////////////////////////////////
debug_vert_output debug_vert_main(a2v a)
{
    debug_vert_output o;
	// Calculate the position of the vertex against the world, view, and projection matrices.
	float4 v = float4(a.inputPosition, 1.0f);
	v = mul(v, viewMatrix);
	o.position = mul(v, projectionMatrix);

    return o;
}
