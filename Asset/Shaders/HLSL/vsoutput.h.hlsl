struct basic_vert_output
{
    float4 pos              : SV_Position;
    float4 normal           : NORMAL0;
    float4 normal_world     : NORMAL1;
    float4 v                : POSITION1;
    float4 v_world          : POSITION2;
    float2 uv               : TEXCOORD0;
};

struct pbr_vert_output
{
    float4 pos              : SV_Position;
    float4 normal           : NORMAL0;
    float4 normal_world     : NORMAL1;
    float4 v                : POSITION1;
    float4 v_world          : POSITION2;
    float2 uv               : TEXCOORD0;
    float3 v_tangent        : POSITION3;
    float3 camPos_tangent   : POSITION4;
    float3x3 TBN            : MATRIX0;
};

struct pos_only_vert_output
{
    float4 pos              : SV_Position;
};

struct cube_vert_output
{
    float4 pos              : SV_Position;
    float3 uvw              : TEXCOORD0;
};

struct simple_vert_output
{
    float4 pos              : SV_Position;
    float2 uv               : TEXCOORD0;
};
