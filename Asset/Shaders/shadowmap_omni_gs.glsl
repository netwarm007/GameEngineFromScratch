layout (triangles) in;
layout (triangle_strip, max_vertices=18) out;

layout(push_constant) uniform gs_constant_t {
    float layer_index;
} u_gsPushConstants;

layout(std140,binding=2) uniform ShadowMatrices {
    mat4 shadowMatrices[6];
};

layout(location = 0) out vec4 FragPos; // FragPos from GS (output per emitvertex)

void main()
{
    for(int face = 0; face < 6; face++)
    {
        gl_Layer = int(u_gsPushConstants.layer_index) * 6 + face; // built-in variable that specifies to which face we render.
        for(int i = 0; i < 3; ++i) // for each triangle's vertices
        {
            FragPos = gl_in[i].gl_Position;
            gl_Position = shadowMatrices[face] * FragPos;
            EmitVertex();
        }    
        EndPrimitive();
    }
}  
