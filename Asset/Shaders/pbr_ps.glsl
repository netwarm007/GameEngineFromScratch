////////////////////////////////////////////////////////////////////////////////
// Filename: pbr_ps.glsl
////////////////////////////////////////////////////////////////////////////////

/////////////////////
// INPUT VARIABLES //
/////////////////////
in vec4 normal;
in vec4 normal_world;
in vec4 v; 
in vec4 v_world;
in vec2 uv;

//////////////////////
// OUTPUT VARIABLES //
//////////////////////
out vec4 outputColor;

////////////////////////////////////////////////////////////////////////////////
// Pixel Shader
////////////////////////////////////////////////////////////////////////////////
void main()
{		
    vec3 N = normalize(normal_world.xyz);
    vec3 V = normalize(camPos - v_world.xyz);

    vec3 albedo;
    float meta = metallic;
    if (usingDiffuseMap)
    {
        albedo = texture(diffuseMap, uv).rgb; 
    }
    else
    {
        albedo = diffuseColor;
    }

    if (usingMetallicMap)
    {
        meta = texture(metallicMap, uv).r; 
    }

    vec3 F0 = vec3(0.04); 
    F0 = mix(F0, albedo, meta);
	           
    // reflectance equation
    vec3 Lo = vec3(0.0);
    for (int i = 0; i < numLights; i++)
    {
        Light light = allLights[i];

        // calculate per-light radiance
        vec3 L = normalize(light.lightPosition.xyz - v_world.xyz);
        vec3 H = normalize(V + L);

        float NdotL = max(dot(N, L), 0.0f);

        // shadow test
        float visibility = shadow_test(v_world, light, NdotL);

        float lightToSurfDist = length(L);
        float lightToSurfAngle = acos(dot(-L, light.lightDirection.xyz));

        // angle attenuation
        float atten = apply_atten_curve(lightToSurfAngle, light.lightAngleAttenCurveParams);

        // distance attenuation
        atten *= apply_atten_curve(lightToSurfDist, light.lightDistAttenCurveParams);

        vec3 radiance = light.lightIntensity * atten * light.lightColor.rgb;
        
        // cook-torrance brdf
        float rough = roughness;
        if (usingRoughnessMap)
        {
            rough = texture(roughnessMap, uv).r; 
        }

        float NDF = DistributionGGX(N, H, rough);        
        float G   = GeometrySmith(N, V, L, rough);      
        vec3 F    = fresnelSchlick(max(dot(H, V), 0.0), F0);       
        
        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= 1.0 - meta;	  
        
        vec3 numerator    = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * NdotL;
        vec3 specular     = numerator / max(denominator, 0.001);  
            
        // add to outgoing radiance Lo
        Lo += (kD * albedo / PI + specular) * radiance * NdotL * visibility; 
    }   
  
    vec3 ambient = ambientColor.rgb;
    {
        float ambientOcc = ao;
        if (usingAoMap)
        {
            ambientOcc = texture(aoMap, uv).r;
        }

        vec3 kS = fresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness);
        vec3 kD = 1.0 - kS;
        vec3 irradiance = textureLod(skybox, N, 1).rgb;
        vec3 diffuse = irradiance * albedo;
        ambient = (kD * diffuse) * ambientOcc;
    }

    vec3 linearColor = ambient + Lo;
	
    // tone mapping
    linearColor = reinhard_tone_mapping(linearColor);
   
    // gamma correction
    linearColor = gamma_correction(linearColor);

    outputColor = vec4(linearColor, 1.0);
}
