#version 450
#define simple_frag_main main

layout(location = 0) in vec3 fragColor;

layout(location = 0) out vec4 outColor;

void simple_frag_main() {
    outColor = vec4(fragColor, 1.0);
}