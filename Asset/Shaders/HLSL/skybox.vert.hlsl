#include "cbuffer.h"
#include "vsoutput.h.hlsl"

cube_vert_output skybox_vert_main(a2v_pos_only a)
{
    cube_vert_output o;
    o.uvw = a.inputPosition.xyz;
    float4x4 _matrix = viewMatrix;
    _matrix[3][0] = 0.0f;
    _matrix[3][1] = 0.0f;
    _matrix[3][2] = 0.0f;
	float4 pos = mul(mul(float4(a.inputPosition, 1.0f), _matrix), projectionMatrix);
    o.pos = pos.xyww;

    return o;
}  