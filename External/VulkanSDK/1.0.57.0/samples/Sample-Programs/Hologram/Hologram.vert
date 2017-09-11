#version 310 es

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_normal;

layout(std140, set = 0, binding = 0) readonly buffer param_block {
	vec3 light_pos;
	vec3 light_color;
	mat4 model;
	mat4 view_projection;
	float alpha;
} params;

layout(location = 0) out vec3 color;
layout(location = 1) out float alpha;

void main()
{
	vec3 world_light = vec3(params.model * vec4(params.light_pos, 1.0));
	vec3 world_pos = vec3(params.model * vec4(in_pos, 1.0));
	vec3 world_normal = mat3(params.model) * in_normal;

	vec3 light_dir = world_light - world_pos;
	float brightness = dot(light_dir, world_normal) / length(light_dir) / length(world_normal);
	brightness = abs(brightness);

	gl_Position = params.view_projection * vec4(world_pos, 1.0);
	color = params.light_color * brightness;
	alpha = params.alpha;
}
