#version 460
layout (triangles) in;

#ifdef FACE_NORMALS

#ifdef VERTEX_NORMALS
    layout (line_strip, max_vertices = 8) out;
#else
    layout (line_strip, max_vertices = 2) out;
#endif

#else
    layout (line_strip, max_vertices = 6) out;
#endif

in VS_OUT {
    vec3 position;
    vec3 normal;
    mat4 mvp;
} gs_in[];

uniform float normal_magnitude;
uniform float face_normal;
/// following https://github.com/mmatl/pyrender/blob/9ddad1da09582b79a2dd647afc4263899331c128/pyrender/shaders/vertex_normals.frag
void GenerateVertNormal(int index)
{

    vec4 p0 = gs_in[index].mvp * vec4(gs_in[index].position, 1.0);
    vec4 p1 = gs_in[index].mvp * vec4(normal_magnitude * normalize(gs_in[index].normal) + gs_in[index].position, 1.0);
    gl_Position = p0;
    EmitVertex();
    gl_Position = p1;
    EmitVertex();
    EndPrimitive();
}

void GenerateFaceNormal()
{
    vec3 p0 = gs_in[0].position.xyz;
    vec3 p1 = gs_in[1].position.xyz;
    vec3 p2 = gs_in[2].position.xyz;

    vec3 v0 = p0 - p1;
    vec3 v1 = p2 - p1;

    vec3 N = normalize(cross(v1, v0));
    vec3 P = (p0 + p1 + p2) / 3.0;

    vec4 np0 = gs_in[0].mvp * vec4(P, 1.0);
    vec4 np1 = gs_in[0].mvp * vec4(normal_magnitude *  N + P, 1.0);

    gl_Position = np0;
    EmitVertex();
    gl_Position = np1;
    EmitVertex();
    EndPrimitive();
}

void main()
{
    if (face_normal > 0)  
      GenerateFaceNormal();
    else {
      GenerateVertNormal(0);
      GenerateVertNormal(1);
      GenerateVertNormal(2);
    }

// #ifdef VERTEX_NORMALS
//    
//#endif

}