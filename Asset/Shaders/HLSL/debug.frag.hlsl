#include "cbuffer.h"
#include "functions.h.hlsl"
#include "vsoutput.h.hlsl"

MYGE_ROOT_SIGNATURE(MyRS1)
float4 debug_frag_main(pos_only_vert_output _entryPointOutput) : SV_Target
{
    return 1.0f.xxxx;
}
