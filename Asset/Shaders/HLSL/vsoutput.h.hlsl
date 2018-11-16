struct vert_output
{
    float4 position         : SV_Position;
    float4 normal           : TEXCOORD0;
    float4 normal_world     : TEXCOORD1;
    float4 v                : TEXCOORD2;
    float4 v_world          : TEXCOORD3;
    float2 uv               : TEXCOORD4;
    float3x3 TBN            : TEXCOORD5;
    float3 v_tangent        : TEXCOORD8;
    float3 camPos_tangent   : TEXCOORD9;
};

struct debug_vert_output
{
    float4 position         : SV_Position;
};
