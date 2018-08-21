////////////////////////////////////////////////////////////////////////////////
// Filename: basic.ps 
////////////////////////////////////////////////////////////////////////////////

/////////////////////
// INPUT VARIABLES //
/////////////////////
in vec4 normal;
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

vec3 projectOnPlane(vec3 point, vec3 center_of_plane, vec3 normal_of_plane)
{
    return point - dot(point - center_of_plane, normal_of_plane) * normal_of_plane;
}

bool isAbovePlane(vec3 point, vec3 center_of_plane, vec3 normal_of_plane)
{
    return dot(point - center_of_plane, normal_of_plane) > 0.0f;
}

vec3 linePlaneIntersect(vec3 line_start, vec3 line_dir, vec3 center_of_plane, vec3 normal_of_plane)
{
    return line_start + line_dir * (dot(center_of_plane - line_start, normal_of_plane) / dot(line_dir, normal_of_plane));
}

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

float shadow_test(const Light light, const float cosTheta) {
    vec4 v_light_space = light.lightVP * v_world;
    v_light_space /= v_light_space.w;

    const mat4 depth_bias = mat4 (
        vec4(0.5f, 0.0f, 0.0f, 0.0f),
        vec4(0.0f, 0.5f, 0.0f, 0.0f),
        vec4(0.0f, 0.0f, 0.5f, 0.0f),
        vec4(0.5f, 0.5f, 0.5f, 1.0f)
    );

    const vec2 poissonDisk[4] = vec2[](
        vec2( -0.94201624, -0.39906216 ),
        vec2( 0.94558609, -0.76890725 ),
        vec2( -0.094184101, -0.92938870 ),
        vec2( 0.34495938, 0.29387760 )
    );

    v_light_space = depth_bias * v_light_space;

    // shadow test
    float visibility = 1.0f;
    if (light.lightShadowMapIndex != -1) // the light cast shadow
    {
        float bias = 5e-4 * tan(acos(cosTheta)); // cosTheta is dot( n,l ), clamped between 0 and 1
        bias = clamp(bias, 0, 0.01);
        for (int i = 0; i < 4; i++)
        {
            float near_occ = texture(shadowMap, vec3(v_light_space.xy + poissonDisk[i] / 700.0f, light.lightShadowMapIndex)).r;
            if (v_light_space.z - near_occ > bias)
            {
                // we are in the shadow
                visibility -= 0.2f;
            }
        }
    }

    return visibility;
}

vec3 apply_light(const Light light) {
    vec3 N = normalize(normal.xyz);
    vec3 L;
    vec3 light_dir = normalize((viewMatrix * worldMatrix * light.lightDirection).xyz);

    if (light.lightPosition.w == 0.0f)
    {
        L = -light_dir;
    }
    else
    {
        L = (viewMatrix * worldMatrix * light.lightPosition).xyz - v.xyz;
    }

    float lightToSurfDist = length(L);

    L = normalize(L);

    float cosTheta = clamp(dot(N, L), 0.0f, 1.0f);

    // shadow test
    float visibility = shadow_test(light, cosTheta);

    float lightToSurfAngle = acos(dot(L, -light_dir));

    // angle attenuation
    float atten_params[5];
    atten_params[0] = light.lightAngleAttenCurveParams_0;
    atten_params[1] = light.lightAngleAttenCurveParams_1;
    atten_params[2] = light.lightAngleAttenCurveParams_2;
    atten_params[3] = light.lightAngleAttenCurveParams_3;
    atten_params[4] = light.lightAngleAttenCurveParams_4;
    float atten = apply_atten_curve(lightToSurfAngle, light.lightAngleAttenCurveType, atten_params);

    // distance attenuation
    atten_params[0] = light.lightDistAttenCurveParams_0;
    atten_params[1] = light.lightDistAttenCurveParams_1;
    atten_params[2] = light.lightDistAttenCurveParams_2;
    atten_params[3] = light.lightDistAttenCurveParams_3;
    atten_params[4] = light.lightDistAttenCurveParams_4;
    atten *= apply_atten_curve(lightToSurfDist, light.lightDistAttenCurveType, atten_params);

    vec3 R = normalize(2.0f * dot(L, N) *  N - L);
    vec3 V = normalize(-v.xyz);

    vec3 linearColor;

    if (usingDiffuseMap)
    {
        linearColor = light.lightIntensity * atten * light.lightColor.rgb * (texture(diffuseMap, uv).rgb * cosTheta + specularColor.rgb * pow(clamp(dot(R, V), 0.0f, 1.0f), specularPower)); 
    }
    else
    {
        linearColor = light.lightIntensity * atten * light.lightColor.rgb * (diffuseColor.rgb * cosTheta + specularColor.rgb * pow(clamp(dot(R, V), 0.0f, 1.0f), specularPower)); 
    }

    return linearColor * visibility;
}

vec3 apply_areaLight(const Light light)
{
    vec3 N = normalize(normal.xyz);
    vec3 right = normalize((viewMatrix * worldMatrix * vec4(1.0f, 0.0f, 0.0f, 0.0f)).xyz);
    vec3 pnormal = normalize((viewMatrix * worldMatrix * light.lightDirection).xyz);
    vec3 ppos = (viewMatrix * worldMatrix * light.lightPosition).xyz;
    vec3 up = normalize(cross(pnormal, right));
    right = normalize(cross(up, pnormal));

    //width and height of the area light:
    float width = light.lightSize.x;
    float height = light.lightSize.y;

    //project onto plane and calculate direction from center to the projection.
    vec3 projection = projectOnPlane(v.xyz, ppos, pnormal);// projection in plane
    vec3 dir = projection - ppos;

    //calculate distance from area:
    vec2 diagonal = vec2(dot(dir,right), dot(dir,up));
    vec2 nearest2D = vec2(clamp(diagonal.x, -width, width), clamp(diagonal.y, -height, height));
    vec3 nearestPointInside = ppos + right * nearest2D.x + up * nearest2D.y;

    vec3 L = nearestPointInside - v.xyz;

    float lightToSurfDist = length(L);
    L = normalize(L);

    // distance attenuation
    float atten_params[5];
    atten_params[0] = light.lightDistAttenCurveParams_0;
    atten_params[1] = light.lightDistAttenCurveParams_1;
    atten_params[2] = light.lightDistAttenCurveParams_2;
    atten_params[3] = light.lightDistAttenCurveParams_3;
    atten_params[4] = light.lightDistAttenCurveParams_4;
    float atten = apply_atten_curve(lightToSurfDist, light.lightDistAttenCurveType, atten_params);

    vec3 linearColor = vec3(0.0f);

    float pnDotL = dot(pnormal, -L);
    float nDotL = dot(N, L);

    if (nDotL > 0.0f && isAbovePlane(v.xyz, ppos, pnormal)) //looking at the plane
    {
        //shoot a ray to calculate specular:
        vec3 V = normalize(-v.xyz);
        vec3 R = normalize(2.0f * dot(V, N) *  N - V);
        vec3 R2 = normalize(2.0f * dot(L, N) *  N - L);
        vec3 E = linePlaneIntersect(v.xyz, R, ppos, pnormal);

        float specAngle = clamp(dot(-R, pnormal), 0.0f, 1.0f);
        vec3 dirSpec = E - ppos;
        vec2 dirSpec2D = vec2(dot(dirSpec, right), dot(dirSpec, up));
        vec2 nearestSpec2D = vec2(clamp(dirSpec2D.x, -width, width), clamp(dirSpec2D.y, -height, height));
        float specFactor = 1.0f - clamp(length(nearestSpec2D - dirSpec2D), 0.0f, 1.0f);

        if (usingDiffuseMap)
        {
            linearColor = light.lightIntensity * atten * light.lightColor.rgb * (texture(diffuseMap, uv).rgb * nDotL * pnDotL + specularColor.rgb * pow(clamp(dot(R2, V), 0.0f, 1.0f), specularPower) * specFactor * specAngle); 
        }
        else
        {
            linearColor = light.lightIntensity * atten * light.lightColor.rgb * (diffuseColor.rgb * nDotL * pnDotL + specularColor.rgb * pow(clamp(dot(R2, V), 0.0f, 1.0f), specularPower) * specFactor * specAngle); 
        }
    }

    return linearColor;
}

void main(void)
{
    vec3 linearColor = vec3(0.0f);
    for (int i = 0; i < numLights; i++)
    {
        if (allLights[i].lightSize.x > 0.0f || allLights[i].lightSize.y > 0.0f)
        {
            linearColor += apply_areaLight(allLights[i]); 
        }
        else
        {
            linearColor += apply_light(allLights[i]); 
        }
    }

    // no-gamma correction
    // outputColor = vec4(clamp(linearColor, 0.0f, 1.0f), 1.0f);

    // gamma correction
    outputColor = vec4(ambientColor.rgb + clamp(pow(linearColor, vec3(1.0f/2.2f)), 0.0f, 1.0f), 1.0f);
}

