/*
 * Vertex shader used by overlay layer
 */
#version 140
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout(location=0) in vec4 pos;
layout(location=1) in vec2 uv_in;

layout(location=0) out vec2 uv;

void main() {
    gl_Position.xy = (pos.xy / 256.0) - 1.0;
    gl_Position.zw = vec2(0, 1);
    uv = uv_in;
}
