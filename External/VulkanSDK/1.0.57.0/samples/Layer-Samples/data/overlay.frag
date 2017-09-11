/*
 * Fragment shader for overlay
 */
#version 140
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (location = 0) in vec2 uv;
layout (set=0, binding=0) uniform sampler2D s;

layout (location = 0) out vec4 uFragColor;
void main() {
    float u = texture(s, uv).x;
    uFragColor.xyzw = vec4(u);

    if (u == 0.0) {
        discard;
    }
}
