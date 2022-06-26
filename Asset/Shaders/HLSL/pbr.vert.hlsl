#include "cbuffer.h"
#include "vsoutput.h.hlsl"

pbr_vert_output pbr_vert_main(a2v a)
{
    pbr_vert_output o;

    o.v_world = mul(float4(a.inputPosition, 1.0f), modelMatrix);
    o.v = mul(o.v_world, viewMatrix);
    o.pos = mul(o.v, projectionMatrix);
    o.normal_world = normalize(mul(float4(a.inputNormal, 0.0f), modelMatrix));
    o.normal = normalize(mul(o.normal_world, viewMatrix));
    float3 tangent = mul(float4(a.inputTangent, 0.0f), modelMatrix).xyz;
    tangent = normalize(tangent - (o.normal_world.xyz * dot(tangent, o.normal_world.xyz)));
    float3 bitangent = cross(o.normal_world.xyz, tangent);
    o.TBN = float3x3(float3(tangent), float3(bitangent), float3(o.normal_world.xyz));
    float3x3 TBN_trans = transpose(o.TBN);
    o.v_tangent = mul(o.v_world.xyz, TBN_trans);
    o.camPos_tangent = mul(camPos.xyz, TBN_trans);
    o.uv.x = a.inputUV.x;
    o.uv.y = 1.0f - a.inputUV.y;

    return o;
}
