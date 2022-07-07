#include "functions.h.hlsl"
#include "vsoutput.h.hlsl"

[RootSignature(MyRS1)] float4 pbr_frag_main(pbr_vert_output _entryPointOutput)
    : SV_Target {
  // offset texture coordinates with Parallax Mapping
  // float3 viewDir   = normalize(_entryPointOutput.camPos_tangent -
  // input.v_tangent); float2 texCoords = ParallaxMapping(_entryPointOutput.uv,
  // viewDir);
  float2 texCoords = _entryPointOutput.uv;

  float3 albedo =
      inverse_gamma_correction(diffuseMap.Sample(samp0, texCoords).rgb);
  float alpha = diffuseMap.Sample(samp0, texCoords).a;
  float meta = metallicMap.Sample(samp0, texCoords).r;
  float rough = metallicMap.Sample(samp0, texCoords).g;
  float3 tangent_normal;
  tangent_normal.xy = normalMap.Sample(samp0, texCoords).rg;
  tangent_normal = tangent_normal * 2.0f - 1.00392f;
  tangent_normal.z = sqrt(clamp(1.0f - tangent_normal.x * tangent_normal.x -
    tangent_normal.y * tangent_normal.y, 0.0f, 1.0f));
  float3 N = mul(tangent_normal, _entryPointOutput.TBN);

  float3 V = normalize(camPos.xyz - _entryPointOutput.v_world.xyz);
  float3 R = reflect(-V, N);

  float3 F0 = 0.04f.xxx;
  F0 = lerp(F0, albedo, meta);

  // reflectance equation
  float3 Lo = 0.0f.xxx;
  for (int i = 0; i < numLights; i++) {
    Light light = lights[i];

    // calculate per-light radiance
    float3 L =
        normalize(light.lightPosition.xyz - _entryPointOutput.v_world.xyz);
    float3 H = normalize(V + L);

    float NdotL = max(dot(N, L), 0.0f);

    // shadow test
    float visibility = shadow_test(_entryPointOutput.v_world, light, NdotL, clip_space_type);

    float lightToSurfDist = length(L);
    float lightToSurfAngle = acos(dot(-L, light.lightDirection.xyz));

    // angle attenuation
    float atten =
        apply_atten_curve(lightToSurfAngle, light.lightAngleAttenCurveType,
                          light.lightAngleAttenCurveParams);

    // distance attenuation
    atten *= apply_atten_curve(lightToSurfDist, light.lightDistAttenCurveType,
                               light.lightDistAttenCurveParams);

    float3 radiance = light.lightIntensity * atten * light.lightColor.rgb;

    // cook-torrance brdf
    float NDF = DistributionGGX(N, H, rough);
    float G = GeometrySmithDirect(N, V, L, rough);
    float3 F = fresnelSchlick(max(dot(H, V), 0.0f), F0);

    float3 kS = F;
    float3 kD = 1.0f.xxx - kS;
    kD *= (1.0f - meta).xxx;

    float3 numerator = NDF * G * F;
    float denominator = 4.0f * max(dot(N, V), 0.0f) * NdotL;
    float3 specular = numerator / max(denominator, 0.001f);

    // add to outgoing radiance Lo
    Lo += (kD * albedo + kS * specular) / PI * radiance * NdotL * visibility;
  }

  float3 ambient;
  {
    // ambient diffuse
    float ambientOcc = metallicMap.Sample(samp0, texCoords).b;

    float3 F = fresnelSchlickRoughness(max(dot(N, V), 0.0f), F0, rough);
    float3 kS = F;
    float3 kD = 1.0f - kS;
    kD *= 1.0f - meta;

    float3 irradiance;
#if defined(OS_WEBASSEMBLY)
    {
      float3 uvw = convert_xyz_to_cube_uv(N);
      irradiance = skybox.SampleLevel(samp0, uvw, 1.0f).rgb;
    }
#else
    irradiance = skybox.SampleLevel(samp0, float4(N, 0.0f), 1.0f).rgb;
#endif
    float3 diffuse = irradiance * albedo;

    // ambient reflect
    const float MAX_REFLECTION_LOD = 9.0f;
    float3 prefilteredColor;
#if defined(OS_WEBASSEMBLY)
    {
      float3 uvw = convert_xyz_to_cube_uv(R);
      prefilteredColor =
          skybox.SampleLevel(samp0, uvw, rough * MAX_REFLECTION_LOD).rgb;
    }
#else
    prefilteredColor =
        skybox.SampleLevel(samp0, float4(R, 1.0f), rough * MAX_REFLECTION_LOD)
            .rgb;
#endif
    float2 envBRDF =
        brdfLUT.Sample(samp0, float2(max(dot(N, V), 0.0f), rough)).rg;
    float3 specular = prefilteredColor * (F * envBRDF.x + envBRDF.y);

    ambient = (kD * diffuse + kS * specular) * ambientOcc;
  }

  float3 linearColor = ambient + Lo;

  // tone mapping
  linearColor = reinhard_tone_mapping(linearColor);

  // gamma correction
  linearColor = gamma_correction(linearColor);

  return float4(linearColor, alpha);
}