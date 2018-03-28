#include "cbuffer.h"
#include "vsoutput.hs"

v2p VSMain(a2v input) {
    v2p output;

	float4 temp = mul(m_viewMatrix, mul(m_worldMatrix, mul(objectMatrix, float4(input.Position.xyz, 1.0f))));
	output.vPosInView = temp.xyz;
	output.Position = mul(m_projectionMatrix, temp);
	float3 vN = mul(m_viewMatrix, mul(m_worldMatrix, mul(objectMatrix, float4(input.Normal, 0.0f)))).xyz;

	output.vNorm = vN;

	output.TextureUV.x = input.TextureUV.x;
	output.TextureUV.y = 1.0f - input.TextureUV.y;

	return output;
}

SamplerState samp0 : register(s0);
Texture2D colorMap : register(t0);
//Texture2D bumpGlossMap: register(t1);

float4 PSMain(v2p input) : SV_TARGET
{
	float3 lightRgb = m_lightColor.xyz;

	const float3 vN = normalize(input.vNorm);
	const float3 vL = normalize(mul(m_viewMatrix, mul(m_worldMatrix, float4(m_lightPosition, 1.0f))).xyz - input.vPosInView);
    const float3 vR = normalize(2 * clamp(dot(vL, vN), 0.0f, 1.0f) * vN - vL);
	const float3 vV = normalize(-input.vPosInView);

	float3 vLightInts; 
	if (usingDiffuseMap)
	{
		vLightInts = ambientColor 
								+ lightRgb * (colorMap.Sample(samp0, input.TextureUV).rgb * clamp(dot(vN, vL), 0.0f, 1.0f) 
								+ specularColor.rgb * pow(clamp(dot(vR,vV), 0.0f, 1.0f), specularPower));
	}
	else{
		vLightInts = ambientColor 
								+ lightRgb * (diffuseColor.rgb * clamp(dot(vN, vL), 0.0f, 1.0f) 
								+ specularColor.rgb * pow(clamp(dot(vR,vV), 0.0f, 1.0f), specularPower));
	}

	return float4(vLightInts, 1.0f);
}

