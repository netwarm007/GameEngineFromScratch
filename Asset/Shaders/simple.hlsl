#include "cbuffer2.h"
#include "vsoutput2.hs"

v2p VSMain(a2v input) {
    v2p output;

	output.Position = mul(mul(mul(float4(input.Position.xyz, 1.0f), m_worldMatrix), m_viewMatrix), m_projectionMatrix);
	float3 vN = (mul(mul(float4(input.Normal, 0.0f), m_worldMatrix), m_viewMatrix)).xyz;
	output.vPosInView = (mul(mul(float4(input.Position.xyz, 1.0f), m_worldMatrix), m_viewMatrix)).xyz;

	output.vNorm = vN;

	//output.TextureUV = input.TextureUV;

	return output;
}

SamplerState samp0 : register(s0);
Texture2D colorMap : register(t0);
//Texture2D bumpGlossMap: register(t1);

float4 PSMain(v2p input) : SV_TARGET
{
	//float3 lightRgb = m_lightColor.xyz;

	//const float3 vN = normalize(input.vNorm);
	//const float3 vL = normalize(m_lightPosition.xyz - input.vPosInView);
    //const float3 vR = normalize(2 * dot(vL, vN) * vN - vL);
	//const float3 vV = normalize(float3(0.0f,0.0f,0.0f) - input.vPosInView);
	//float d = length(vL); 

	//float3 vLightInts = ambientColor + lightRgb * diffuseColor * dot(vN, vL) + specularColor * pow(clamp(dot(vR,vV), 0.0f, 1.0f), specularPower);

	//return float4(vLightInts, 1.0f);
	return float4(1.0f, 1.0f, 1.0f, 1.0f);
}

