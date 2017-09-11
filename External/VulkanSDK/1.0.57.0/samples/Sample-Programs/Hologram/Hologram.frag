#version 310 es

precision highp float;

layout(location = 0) in vec3 color;
layout(location = 1) in float alpha;

layout(location = 0) out vec4 fragcolor;

void main()
{
	fragcolor = vec4(color, alpha);
}
