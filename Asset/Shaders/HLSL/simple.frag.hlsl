#include "vsoutput.h.hlsl"

Texture2D tex : register(t1);
SamplerState samp0 : register(s1);

#define MyRS1                                           \
    "RootFlags( ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT ), " \
    "DescriptorTable(CBV(b0, numDescriptors = 1, "    \
    "        flags = DESCRIPTORS_VOLATILE), "           \
    "SRV(t1, numDescriptors = 8, "     \
    "        flags = DESCRIPTORS_VOLATILE)), "         \
    "DescriptorTable( Sampler(s1, space=0, numDescriptors = 8))"

[RootSignature(MyRS1)]
float4 simple_frag_main(simple_vert_output _entryPointOutput) : SV_Target
{
    return tex.Sample(samp0, _entryPointOutput.uv);
}
