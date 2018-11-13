layout(vertices = 4) out;

void main(){
     #define id gl_InvocationID
     gl_out[id].gl_Position = gl_in[id].gl_Position;
     if(id == 0){
         vec4 v0 = project(gl_in[0].gl_Position);
         vec4 v1 = project(gl_in[1].gl_Position);
         vec4 v2 = project(gl_in[2].gl_Position);
         vec4 v3 = project(gl_in[3].gl_Position);

         if(all(bvec4(
             offscreen(v0),
             offscreen(v1),
             offscreen(v2),
             offscreen(v3)
         ))){
             gl_TessLevelInner[0] = 0;
             gl_TessLevelInner[1] = 0;
             gl_TessLevelOuter[0] = 0;
             gl_TessLevelOuter[1] = 0;
             gl_TessLevelOuter[2] = 0;
             gl_TessLevelOuter[3] = 0;
         }
         else{
             vec2 ss0 = screen_space(v0);
             vec2 ss1 = screen_space(v1);
             vec2 ss2 = screen_space(v2);
             vec2 ss3 = screen_space(v3);

             float e0 = level(ss1, ss2);
             float e1 = level(ss0, ss1);
             float e2 = level(ss3, ss0);
             float e3 = level(ss2, ss3);

             gl_TessLevelInner[0] = mix(e1, e2, 0.5);
             gl_TessLevelInner[1] = mix(e0, e3, 0.5);
             gl_TessLevelOuter[0] = e0;
             gl_TessLevelOuter[1] = e1;
             gl_TessLevelOuter[2] = e2;
             gl_TessLevelOuter[3] = e3;
         }
     }
 } 