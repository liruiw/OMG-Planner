#version 460
uniform sampler2D texUnit;
in vec2 theCoords;
in vec3 Normal;
in vec3 Normal_cam;
in vec3 FragPos;
in vec3 Instance_color;
in vec3 Pos_cam;
in vec3 Pos_obj;
in float inverse_normal;
layout (location = 0) out vec4 outputColour;
uniform vec3 normal_color; 
/// following https://github.com/mmatl/pyrender/blob/9ddad1da09582b79a2dd647afc4263899331c128/pyrender/shaders/vertex_normals.frag
void main() {
    outputColour  = vec4(normal_color,1.0);
}