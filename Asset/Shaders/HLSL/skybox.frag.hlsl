#include "cbuffer.h"
#include "functions.h.hlsl"
#include "vsoutput.h.hlsl"

float4 skybox_frag_main(cube_vert_output _entryPointOutput) : SV_TARGET
{
    float4 outputColor;

    outputColor = skybox.SampleLevel(samp0, float4(_entryPointOutput.uvw, 0), 0);

    // tone mapping
    //outputColor.rgb = reinhard_tone_mapping(outputColor.rgb);
    outputColor.rgb = exposure_tone_mapping(outputColor.rgb);

    // gamma correction
    outputColor.rgb = gamma_correction(outputColor.rgb);

    return outputColor;
}