#version 330 core
// Mask an image using another image
//#define SHADERTOY

// Output color
out vec4 p3d_Color;

uniform vec2 iResolution;
uniform sampler2D iChannel0;
uniform sampler2D iChannel1;


void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 rec_res = 1.0 / iResolution.xy;
    vec2 p = fragCoord.xy * rec_res;
    if (texture(iChannel0, p).x == 0)
    {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    fragColor = texture(iChannel1, p);

    if (fragColor.rgb == vec3(0.0, 0.0, 0.0)) {
        fragColor = vec4(0.1, 0.5, 0.1, 1.0);
    }
}

#ifndef SHADERTOY
void main(){
    mainImage( p3d_Color, gl_FragCoord.xy );
}
#endif