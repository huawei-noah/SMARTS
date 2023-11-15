#version 330 core
// Visibility check

// -----------------------------------------------
#define STEP_LENGTH 0.2
#define CENTER_COORD iResolution.xy * 0.5
#define DEVICE_HEIGHT 0.8
#define TOPOLOGY_SCALING_FACTOR 1.0
#define CENTER vec2(0.5)
//#define SHADERTOY

#ifdef SHADERTOY
float get_texture_max( vec2 uv ) {
   return texture(iChannel0, uv).x;
}
#else
// Output color
out vec4 p3d_Color;

uniform vec2 iResolution;
uniform sampler2D iChannel0;
uniform sampler2D iChannel1;
uniform sampler2D iChannel2;

float get_texture_max( vec2 uv ) {
   return texture(iChannel0, uv).x + texture(iChannel1, uv).x;
}
#endif

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 rec_res = 1.0 / iResolution.xy;
    vec2 p = fragCoord.xy * rec_res;

    vec2 offset = fragCoord.xy - CENTER_COORD;
    vec2 offset_norm = normalize(offset);
    float offset_dist = length(offset);
    
    if (texture(iChannel2, p).x <= 0.0){
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    
    float center_height = get_texture_max(vec2(0.5)) * TOPOLOGY_SCALING_FACTOR + DEVICE_HEIGHT;
    float target_height = get_texture_max(p) * TOPOLOGY_SCALING_FACTOR;

    float target_slope = (target_height - center_height) / offset_dist;
    
    float total = STEP_LENGTH;
    while (total < offset_dist - 0.01 ) {
        vec2 intermediary_coord = (CENTER_COORD + offset_norm * total) * rec_res;
        float intermediary_height = get_texture_max(intermediary_coord) * TOPOLOGY_SCALING_FACTOR;
        
        if ( target_slope < ((intermediary_height - center_height) * TOPOLOGY_SCALING_FACTOR ) / total){
            fragColor = vec4(0.0, 0.0, 0.0, 1.0);
            return;
        }
        
        
        total += STEP_LENGTH;
    }
    

    fragColor = vec4(1.0,1.0,1.0,1.0);
}

#ifndef SHADERTOY
void main(){
    mainImage( p3d_Color, gl_FragCoord.xy );
}
#endif