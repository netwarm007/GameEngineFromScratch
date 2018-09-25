layout(location = 0) in vec3 UVW;

layout(location = 0) out vec3 color;

layout(push_constant) uniform debugPushConstants {
    float level;
} u_pushConstants;

layout(binding = 0) uniform samplerCube depthSampler;

void main(){
    color = textureLod(depthSampler, UVW, u_pushConstants.level).rgb;
}
