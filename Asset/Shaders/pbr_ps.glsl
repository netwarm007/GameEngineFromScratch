////////////////////////////////////////////////////////////////////////////////
// Filename: pbr_ps.glsl
////////////////////////////////////////////////////////////////////////////////

/////////////////////
// INPUT VARIABLES //
/////////////////////
layout(location = 0) in vec4 normal;
layout(location = 1) in vec4 normal_world;
layout(location = 2) in vec4 v; 
layout(location = 3) in vec4 v_world;
layout(location = 4) in vec2 uv;
layout(location = 5) in mat3 TBN;
layout(location = 8) in vec3 v_tangent;
layout(location = 9) in vec3 camPos_tangent;

//////////////////////
// OUTPUT VARIABLES //
//////////////////////
layout(location = 0) out vec4 outputColor;

////////////////////////////////////////////////////////////////////////////////
// Pixel Shader
////////////////////////////////////////////////////////////////////////////////
void main()
{		
    // offset texture coordinates with Parallax Mapping
    vec3 viewDir   = normalize(camPos_tangent - v_tangent);
    vec2 texCoords = ParallaxMapping(uv, viewDir);

    vec3 tangent_normal = texture(normalMap, texCoords).rgb;
    tangent_normal = tangent_normal * 2.0f - 1.0f;   
    vec3 N = normalize(TBN * tangent_normal); 

    vec3 V = normalize(camPos.xyz - v_world.xyz);
    vec3 R = reflect(-V, N);   

    vec3 albedo = inverse_gamma_correction(texture(diffuseMap, texCoords).rgb); 

    float meta = texture(metallicMap, texCoords).r; 

    float rough = texture(roughnessMap, texCoords).r; 

    vec3 F0 = vec3(0.04f); 
    F0 = mix(F0, albedo, meta);
	           
    // reflectance equation
    vec3 Lo = vec3(0.0f);
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
        float atten = apply_atten_curve(lightToSurfAngle, light.lightAngleAttenCurveType, light.lightAngleAttenCurveParams);

        // distance attenuation
        atten *= apply_atten_curve(lightToSurfDist, light.lightDistAttenCurveType, light.lightDistAttenCurveParams);

        vec3 radiance = light.lightIntensity * atten * light.lightColor.rgb;
        
        // cook-torrance brdf
        float NDF = DistributionGGX(N, H, rough);        
        float G   = GeometrySmithDirect(N, V, L, rough);      
        vec3 F    = fresnelSchlick(max(dot(H, V), 0.0f), F0);       
        
        vec3 kS = F;
        vec3 kD = vec3(1.0f) - kS;
        kD *= 1.0f - meta;	  
        
        vec3 numerator    = NDF * G * F;
        float denominator = 4.0f * max(dot(N, V), 0.0f) * NdotL;
        vec3 specular     = numerator / max(denominator, 0.001f);  
            
        // add to outgoing radiance Lo
        Lo += (kD * albedo / PI + specular) * radiance * NdotL * visibility; 
    }   
  
    vec3 ambient;
    {
        // ambient diffuse
        float ambientOcc = texture(aoMap, texCoords).r;

        vec3 F = fresnelSchlickRoughness(max(dot(N, V), 0.0f), F0, rough);
        vec3 kS = F;
        vec3 kD = 1.0f - kS;
        kD *= 1.0f - meta;	  

        vec3 irradiance = textureLod(skybox, vec4(N, 0.0f), 1.0f).rgb;
        vec3 diffuse = irradiance * albedo;

        // ambient reflect
        const float MAX_REFLECTION_LOD = 9.0f;
        vec3 prefilteredColor = textureLod(skybox, vec4(R, 1.0f), rough * MAX_REFLECTION_LOD).rgb;    
        vec2 envBRDF  = texture(brdfLUT, vec2(max(dot(N, V), 0.0f), rough)).rg;
        vec3 specular = prefilteredColor * (F * envBRDF.x + envBRDF.y);

        ambient = (kD * diffuse + specular) * ambientOcc;
    }

    vec3 linearColor = ambient + Lo;
	
    // tone mapping
    linearColor = reinhard_tone_mapping(linearColor);
   
    // gamma correction
    linearColor = gamma_correction(linearColor);

    outputColor = vec4(linearColor, 1.0f);
}
