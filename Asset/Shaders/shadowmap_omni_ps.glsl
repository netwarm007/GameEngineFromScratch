layout(location = 0) in vec4 FragPos;

layout(push_constant) uniform ps_constant_t {
    vec3 lightPos;
    float far_plane;
} u_lightParams;

void main()
{
    // get distance between fragment and light source
    float lightDistance = length(FragPos.xyz - u_lightParams.lightPos);
    
    // map to [0;1] range by dividing by far_plane
    lightDistance = lightDistance / u_lightParams.far_plane;
    
    // write this as modified depth
    gl_FragDepth = lightDistance;
}  