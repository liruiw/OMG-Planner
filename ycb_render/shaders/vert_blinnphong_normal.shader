#version 460
uniform mat4 V;
uniform mat4 P;
uniform mat4 pose_rot;
uniform mat4 pose_trans;
uniform vec3 instance_color; 
        
layout (location=0) in vec3 position;
layout (location=1) in vec3 normal;
layout (location=2) in vec2 texCoords;
out vec2 theCoords;
out vec3 Normal;
out vec3 FragPos;
out vec3 Normal_cam;
out vec3 Instance_color;
out vec3 Pos_cam;
out vec3 Pos_obj;
out float inverse_normal;
out VS_OUT {
    vec3 position;
    vec3 normal;
    mat4 mvp;
} vs_out;
/// following https://github.com/mmatl/pyrender/blob/9ddad1da09582b79a2dd647afc4263899331c128/pyrender/shaders/vertex_normals.frag
void main() {
  
    vs_out.mvp = P * V * pose_trans * pose_rot;
    vs_out.position = position;
    vs_out.normal = normal;    
}