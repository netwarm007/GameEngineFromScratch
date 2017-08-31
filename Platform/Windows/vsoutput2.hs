#ifndef __VSOUTPUT_H__
#define __VSOUTPUT_H__

struct v2p
{
    float4 Position     : SV_POSITION;
    float2 TextureUV    : TEXCOORD0;
	float3 vNorm		: TEXCOORD1;
    float4 vTang		: TEXCOORD2;
	float3 vPosInView	: TEXCOORD3;
};



#endif

