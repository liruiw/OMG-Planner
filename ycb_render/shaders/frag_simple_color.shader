#version 460
in vec3 theColor;
layout (location = 0) out vec4 outputColour;
layout (location = 1) out vec4 NormalColour;
layout (location = 2) out vec4 InstanceColour;
layout (location = 3) out vec4 PCColour;

void main() {
    outputColour = vec4(theColor,1);
    NormalColour = vec4(theColor,1);
    InstanceColour = vec4(theColor,1);
    PCColour = vec4(theColor,1);

}