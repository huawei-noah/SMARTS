#version 330 core

out vec4 FragColor;

flat in vec3 StartPos;
in vec3 VertPos;

uniform vec4 p3d_Color;
uniform vec2 Resolution;

void main()
{
    uint pattern = 0x00FFu;
    float factor = 2.0;

    vec2 dir = (VertPos.xy - StartPos.xy) * Resolution / 2.0;
    float dist = length(dir);

    uint bit = uint(round(dist / factor)) & 15U;
    if ((pattern & (1U<<bit)) == 0U)
        discard; 
    FragColor = p3d_Color;
}