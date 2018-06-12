#include "cbuffer.h"
#include "vsoutput.hs"

v2p VSMain(a2v input) {
    v2p output;

	float4 temp = mul(m_viewMatrix, mul(m_worldMatrix, mul(objectMatrix, float4(input.Position.xyz, 1.0f))));
	output.vPosInView = temp;
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

 float linear_interpolate(float t, float begin, float end)
{
    if (t < begin)
    {
        return 1.0f;
    }
    else if (t > end)
    {
        return 0.0f;
    }
    else
    {
        return (end - t) / (end - begin);
    }
}

float apply_atten_curve(float dist, int atten_type, float atten_params[5])
{
    float atten = 1.0f;

    switch(atten_type)
    {
        case 1: // linear
        {
            float begin_atten = atten_params[0];
            float end_atten = atten_params[1];
            atten = linear_interpolate(dist, begin_atten, end_atten);
            break;
        }
        case 2: // smooth
        {
            float begin_atten = atten_params[0];
            float end_atten = atten_params[1];
            float tmp = linear_interpolate(dist, begin_atten, end_atten);
            atten = 3.0f * pow(tmp, 2.0f) - 2.0f * pow(tmp, 3.0f);
            break;
        }
        case 3: // inverse
        {
            float scale = atten_params[0];
            float offset = atten_params[1];
            float kl = atten_params[2];
            float kc = atten_params[3];
            atten = clamp(scale / 
                (kl * dist + kc * scale) + offset, 
                0.0f, 1.0f);
            break;
        }
        case 4: // inverse square
        {
            float scale = atten_params[0];
            float offset = atten_params[1];
            float kq = atten_params[2];
            float kl = atten_params[3];
            float kc = atten_params[4];
            atten = clamp(pow(scale, 2.0f) / 
                (kq * pow(dist, 2.0f) + kl * dist * scale + kc * pow(scale, 2.0f) + offset), 
                0.0f, 1.0f);
            break;
        }
        case 0:
        default:
            break; // no attenuation
    }

    return atten;
}

float3 apply_light(v2p input, Light light) {
    float3 N = normalize(input.vNorm);
    float3 L;
    float3 light_dir = normalize(mul(m_viewMatrix, mul(m_worldMatrix, light.m_lightDirection))).xyz;

    if (light.m_lightPosition.w == 0.0f)
    {
        L = -light_dir;
    }
    else
    {
		L = mul(m_viewMatrix, mul(m_worldMatrix, light.m_lightPosition)).xyz - input.vPosInView.xyz;
    }

    float lightToSurfDist = length(L);
    L = normalize(L);
    float lightToSurfAngle = acos(dot(L, light_dir));

    // angle attenuation
    float atten = apply_atten_curve(lightToSurfAngle, light.m_lightAngleAttenCurveType, light.m_lightAngleAttenCurveParams);

    // distance attenuation
    atten *= apply_atten_curve(lightToSurfDist, light.m_lightDistAttenCurveType, light.m_lightDistAttenCurveParams);

    float3 R = normalize(2.0f * clamp(dot(L, N), 0.0f, 1.0f) *  N - L);
    float3 V = normalize(-input.vPosInView.xyz);

    float3 linearColor;

    if (usingDiffuseMap)
    {
        linearColor = ambientColor.rgb + /* light.m_lightIntensity * */ atten * light.m_lightColor.rgb * (colorMap.Sample(samp0, input.TextureUV).rgb * clamp(dot(N, L), 0.0f, 1.0f) + specularColor.rgb * pow(clamp(dot(R, V), 0.0f, 1.0f), specularPower)); 
    }
    else
    {
        linearColor = ambientColor.rgb + /* light.m_lightIntensity * */ atten * light.m_lightColor.rgb * (diffuseColor.rgb * clamp(dot(N, L), 0.0f, 1.0f) + specularColor.rgb * pow(clamp(dot(R, V), 0.0f, 1.0f), specularPower)); 
    }

    return linearColor;
}

float4 PSMain(v2p input) : SV_TARGET
{
    float3 linearColor = 0.0f;
    for (uint i = 0; i < numLights; i++)
    {
        linearColor += apply_light(input, m_lights[i]); 
    }

    // gama correction
    return float4(clamp(pow(linearColor, 1.0f/2.2f), 0.0f, 1.0f), 1.0f);
}

float4 PSMain2(v2p input) : SV_TARGET
{
	float3 lightRgb = m_lights[0].m_lightColor.rgb;

	const float3 vN = normalize(input.vNorm);
	const float3 vL = normalize(mul(m_viewMatrix, mul(m_worldMatrix, m_lights[0].m_lightPosition)).xyz - input.vPosInView.xyz);
    const float3 vR = normalize(2 * clamp(dot(vL, vN), 0.0f, 1.0f) * vN - vL);
	const float3 vV = normalize(-input.vPosInView.xyz);

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

