#include "cbuffer2.h"
#include "vsoutput2.hs"
#include "illum.hs"

v2p VSMain(a2v input) {
    v2p output;

	output.Position = mul(float4(input.Position.xyz, 1), m_modelViewProjection);
	float3 vN = normalize(mul(float4(input.Normal, 0), m_modelView).xyz);
	float3 vT = normalize(mul(float4(input.Tangent.xyz, 0), m_modelView).xyz);
	output.vPosInView = mul(float4(input.Position.xyz, 1), m_modelView).xyz;

	output.vNorm = vN;
	output.vTang = float4(vT, input.Tangent.w);

	output.TextureUV = input.TextureUV;

	return output;
}

SamplerState samp0 : register(s0);
Texture2D colorMap : register(t0);
//Texture2D bumpGlossMap: register(t1);

float4 PSMain(v2p input) : SV_TARGET
{
	float3 lightRgb = m_lightColor.xyz;
	float4 lightAtten = m_lightAttenuation;
	float3 ambientRgb = m_ambientColor.rgb;
	float  specPow = 30;

	const float3 vN = input.vNorm;
	const float3 vT = input.vTang.xyz;
	const float3 vB = input.vTang.w * cross(vN, vT);
	float3 vL = m_lightPosition.xyz - input.vPosInView;
	const float3 vV = normalize(float3(0,0,0) - input.vPosInView);
	float d = length(vL); vL = normalize(vL);
	float attenuation = saturate(1.0f/(lightAtten.x + lightAtten.y * d + lightAtten.z * d * d) - lightAtten.w);

	//float4 normalGloss = bumpGlossMap.Sample(samp0, input.TextureUV.xy);
	float4 normalGloss = { 1.0f, 0.2f, 0.2f, 0.0f };
	normalGloss.xyz = normalGloss.xyz * 2.0f - 1.0f;
	normalGloss.y = -normalGloss.y; // normal map has green channel inverted

	float3 vBumpNorm = normalize(normalGloss.x * vT + normalGloss.y * vB + normalGloss.z * vN);
	float3 vGeomNorm = normalize(vN);

	float3 diff_col = colorMap.Sample(samp0, input.TextureUV.xy).xyz;
	float3 spec_col = 0.4 * normalGloss.w + 0.1;
	float3 vLightInts = attenuation * lightRgb * BRDF2_ts_nphong_nofr(vBumpNorm, vGeomNorm, vL, vV, diff_col, spec_col, specPow);
	vLightInts += (diff_col * ambientRgb);

	return float4(vLightInts, 1);
}

