layout(location = 0) in vec3 UVW;

layout(location = 0) out vec3 color;

layout(push_constant) uniform debugPushConstants {
    float level;
    float layer_index;
} u_pushConstants;

layout(binding = 0) uniform samplerCubeArray depthSampler;

void main(){
    color = textureLod(depthSampler, vec4(UVW, u_pushConstants.layer_index), u_pushConstants.level).rgb;
}