// Input vertex data, different for all executions of this shader.
layout(location = 0) in vec3 inputPosition;

// update per frame
layout(push_constant) uniform constants_t {
	mat4 depthVP;
} u_pushConstants;

void main(){
	gl_Position = u_pushConstants.depthVP * modelMatrix * vec4(inputPosition, 1.0f);
}
