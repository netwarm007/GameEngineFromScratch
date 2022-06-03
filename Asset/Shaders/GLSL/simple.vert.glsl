#version 450
#define simple_vert_main main

layout(location = 0) in  vec2 inPosition;
layout(location = 1) in  vec3 inColor;

layout(location = 0) out vec3 fragColor;

void simple_vert_main () {
    gl_Position = vec4(inPosition, 0.0, 1.0);
    fragColor = inColor;
}