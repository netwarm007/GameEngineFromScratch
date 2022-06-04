#version 450
#define simple_frag_main main

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

layout(binding = 1) uniform sampler2D texSampler;

void simple_frag_main() {
    outColor = texture(texSampler, fragTexCoord);
}