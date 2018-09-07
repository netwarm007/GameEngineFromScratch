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

float apply_atten_curve(float dist, mat4 atten_params)
{
    float atten = 1.0f;

    switch(int(atten_params[0][0]))
    {
        case 1: // linear
        {
            float begin_atten = atten_params[0][1];
            float end_atten = atten_params[0][2];
            atten = linear_interpolate(dist, begin_atten, end_atten);
            break;
        }
        case 2: // smooth
        {
            float begin_atten = atten_params[0][1];
            float end_atten = atten_params[0][2];
            float tmp = linear_interpolate(dist, begin_atten, end_atten);
            atten = 3.0f * pow(tmp, 2.0f) - 2.0f * pow(tmp, 3.0f);
            break;
        }
        case 3: // inverse
        {
            float scale = atten_params[0][1];
            float offset = atten_params[0][2];
            float kl = atten_params[0][3];
            float kc = atten_params[1][0];
            atten = clamp(scale / 
                (kl * dist + kc * scale) + offset, 
                0.0f, 1.0f);
            break;
        }
        case 4: // inverse square
        {
            float scale = atten_params[0][1];
            float offset = atten_params[0][2];
            float kq = atten_params[0][3];
            float kl = atten_params[1][0];
            float kc = atten_params[1][1];
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

vec3 reinhard_tone_mapping(vec3 color)
{
    return color / (color + vec3(1.0f));
}

vec3 exposure_tone_mapping(vec3 color)
{
    const float exposure = 1.0f;
    return vec3(1.0f) - exp(-color * exposure);
}

vec3 gamma_correction(vec3 color)
{
    const float gamma = 2.2f;
    return pow(color, vec3(1.0f / gamma));
}

vec3 inverse_gamma_correction(vec3 color)
{
    const float gamma = 2.2f;
    return pow(color, vec3(gamma));
}
