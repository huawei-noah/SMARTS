#version 330 core

uniform mat4 p3d_ModelViewProjectionMatrix;

in vec4 p3d_Vertex;

flat out vec3 startPos;
out vec3 vertPos;

void main()
{
    vec4 pos = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    gl_Position = pos;
    vertPos = pos.xyz / pos.w;
    startPos = vertPos;
}