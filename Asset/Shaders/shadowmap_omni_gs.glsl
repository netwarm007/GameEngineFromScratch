layout (triangles) in;
layout (triangle_strip, max_vertices=18) out;

layout(push_constant) uniform constants_t {
    vec3 lightPos;
    float far_plane;
    mat4 shadowMatrices[6];
    float layer_index;
} u_pushConstants;

layout(location = 0) out vec4 FragPos; // FragPos from GS (output per emitvertex)

void main()
{
    for(int face = 0; face < 6; face++)
    {
        gl_Layer = int(u_pushConstants.layer_index) * 6 + face; // built-in variable that specifies to which face we render.
        for(int i = 0; i < 3; ++i) // for each triangle's vertices
        {
            FragPos = gl_in[i].gl_Position;
            gl_Position = u_pushConstants.shadowMatrices[face] * FragPos;
            EmitVertex();
        }    
        EndPrimitive();
    }
}  
