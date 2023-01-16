#version 330 core

uniform mat4 p3d_ModelViewProjectionMatrix;

in vec4 p3d_Vertex;

flat out vec3 StartPos;
out vec3 VertPos;

void main()
{
    vec4 pos = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    gl_Position = pos;
    VertPos = pos.xyz / pos.w;
    StartPos = VertPos;
}