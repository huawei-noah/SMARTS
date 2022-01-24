#version 330 core

// Output color
out vec4 FragColor;

// Uniform input: flat color
uniform vec4 p3d_Color;

void main()
{
    FragColor = p3d_Color;
}