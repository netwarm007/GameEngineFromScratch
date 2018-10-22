layout(location = 0) out vec4 outputColor;

layout(location = 1) in vec4 normal_world;
layout(location = 3) in vec4 v_world;
layout(location = 4) in vec2 uv;
layout(location = 5) in mat3 TBN;
layout(location = 8) in vec3 v_tangent;
layout(location = 9) in vec3 camPos_tangent;

void main(){
    vec3 Lo = vec3(0.0f);

    for (int i = 0; i < numLights; i++)
    {
        Light light = allLights[i];

        vec3 L = normalize(light.lightPosition.xyz - v_world.xyz);
        vec3 N = normal_world.xyz;

        float NdotL = max(dot(N, L), 0.0f);

        #if 0
        // shadow test
        float visibility = shadow_test(v_world, light, NdotL);
        #else
        float visibility = 1.0f;
        #endif

        float lightToSurfDist = length(L);
        float lightToSurfAngle = acos(dot(-L, light.lightDirection.xyz));

        // angle attenuation
        float atten = apply_atten_curve(lightToSurfAngle, light.lightAngleAttenCurveType, light.lightAngleAttenCurveParams);

        // distance attenuation
        atten *= apply_atten_curve(lightToSurfDist, light.lightDistAttenCurveType, light.lightDistAttenCurveParams);

        vec3 radiance = light.lightIntensity * atten * light.lightColor.rgb;
        
        // add to outgoing radiance Lo
        Lo += radiance * NdotL * visibility; 
    }

    outputColor = vec4(Lo, 1.0f);
}
