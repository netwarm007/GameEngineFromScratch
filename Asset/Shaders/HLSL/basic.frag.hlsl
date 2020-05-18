#include "functions.h.hlsl"
#include "vsoutput.h.hlsl"

float3 apply_light(const Light light, basic_vert_output input) {
    float3 N = normalize(input.normal.xyz);
    float3 L;
    float3 light_dir = normalize(mul(light.lightDirection, viewMatrix).xyz);

    if (light.lightPosition.w == 0.0f)
    {
        L = -light_dir;
    }
    else
    {
        L = mul(light.lightPosition, viewMatrix).xyz - input.v.xyz;
    }

    float lightToSurfDist = length(L);

    L = normalize(L);

    float cosTheta = clamp(dot(N, L), 0.0f, 1.0f);

    // shadow test
    //float visibility = shadow_test(input.v_world, light, cosTheta);
    float visibility = 1.0f;

    float lightToSurfAngle = acos(dot(L, -light_dir));

    // angle attenuation
    float atten = apply_atten_curve(lightToSurfAngle, light.lightAngleAttenCurveType, light.lightAngleAttenCurveParams);

    // distance attenuation
    atten *= apply_atten_curve(lightToSurfDist, light.lightDistAttenCurveType, light.lightDistAttenCurveParams);

    float3 R = normalize(2.0f * dot(L, N) *  N - L);
    float3 V = normalize(-input.v.xyz);

    float3 linearColor;

    float3 admit_light = light.lightIntensity * atten * light.lightColor.rgb;
    linearColor = diffuseMap.Sample(samp0, input.uv).rgb * cosTheta; 
    if (visibility > 0.2f)
    {
        linearColor += 0.8f * pow(clamp(dot(R, V), 0.0f, 1.0f), 50.0f); 
    }
    linearColor *= admit_light;

    return linearColor * visibility;
}

float3 apply_areaLight(const Light light, basic_vert_output input)
{
    float3 N = normalize(input.normal.xyz);
    float3 right = normalize(mul(float4(1.0f, 0.0f, 0.0f, 0.0f), viewMatrix).xyz);
    float3 pnormal = normalize(mul(light.lightDirection, viewMatrix).xyz);
    float3 ppos = mul(light.lightPosition, viewMatrix).xyz;
    float3 up = normalize(cross(pnormal, right));
    right = normalize(cross(up, pnormal));

    //width and height of the area light:
    float width = light.lightSize.x;
    float height = light.lightSize.y;

    //project onto plane and calculate direction from center to the projection.
    float3 projection = projectOnPlane(input.v.xyz, ppos, pnormal);// projection in plane
    float3 dir = projection - ppos;

    //calculate distance from area:
    float2 diagonal = float2(dot(dir,right), dot(dir,up));
    float2 nearest2D = float2(clamp(diagonal.x, -width, width), clamp(diagonal.y, -height, height));
    float3 nearestPointInside = ppos + right * nearest2D.x + up * nearest2D.y;

    float3 L = nearestPointInside - input.v.xyz;

    float lightToSurfDist = length(L);
    L = normalize(L);

    // distance attenuation
    float atten = apply_atten_curve(lightToSurfDist, light.lightDistAttenCurveType, light.lightDistAttenCurveParams);

    float3 linearColor = 0.0f;

    float pnDotL = dot(pnormal, -L);
    float nDotL = dot(N, L);

    if (nDotL > 0.0f && isAbovePlane(input.v.xyz, ppos, pnormal)) //looking at the plane
    {
        //shoot a ray to calculate specular:
        float3 V = normalize(-input.v.xyz);
        float3 R = normalize(2.0f * dot(V, N) *  N - V);
        float3 R2 = normalize(2.0f * dot(L, N) *  N - L);
        float3 E = linePlaneIntersect(input.v.xyz, R, ppos, pnormal);

        float specAngle = clamp(dot(-R, pnormal), 0.0f, 1.0f);
        float3 dirSpec = E - ppos;
        float2 dirSpec2D = float2(dot(dirSpec, right), dot(dirSpec, up));
        float2 nearestSpec2D = float2(clamp(dirSpec2D.x, -width, width), clamp(dirSpec2D.y, -height, height));
        float specFactor = 1.0f - clamp(length(nearestSpec2D - dirSpec2D), 0.0f, 1.0f);

        float3 admit_light = light.lightIntensity * atten * light.lightColor.rgb;

        linearColor = diffuseMap.Sample(samp0, input.uv).rgb * nDotL * pnDotL; 
        linearColor += 0.8f.xxx * pow(clamp(dot(R2, V), 0.0f, 1.0f), 50.0f) * specFactor * specAngle; 
        linearColor *= admit_light;
    }

    return linearColor;
}

[RootSignature(MyRS1)]
float4 basic_frag_main(basic_vert_output _entryPointOutput) : SV_Target
{
    float3 linearColor = 0.0f.xxx;
    for (int i = 0; i < numLights; i++)
    {
        if (lights[i].lightType == 3) // area light
        {
            linearColor += apply_areaLight(lights[i], _entryPointOutput); 
        }
        else
        {
            linearColor += apply_light(lights[i], _entryPointOutput); 
        }
    }

    // add ambient color
    // linearColor += ambientColor.rgb;
    linearColor += skybox.SampleLevel(samp0, float4(_entryPointOutput.normal_world.xyz, 0), 2.0).rgb * 0.20f.xxx;

    // tone mapping
    //linearColor = reinhard_tone_mapping(linearColor);
    linearColor = exposure_tone_mapping(linearColor);

    // gamma correction
    return float4(linearColor, 1.0f);
}
