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
uniform float iElevation;
uniform sampler2D iChannel0;
uniform sampler2D iChannel1;

float get_texture_max( vec2 uv ) {
   return texture(iChannel0, uv).x * TOPOLOGY_SCALING_FACTOR + texture(iChannel1, uv).x;
}
#endif

float cross_magnitude(vec2 lp1, vec2 lp2, vec2 p1){
    vec2 v1 = lp2 - lp1;
    vec2 v2 = lp2 - p1;
    
    return v1.x * v2.y - v1.y * v2.x;
}

float to_surface_face(vec2 rec_res, vec2 p, float aspect, float elevation) {
    // Determine the minimum delta needed to determine the facing direction of the surface
    float interpolate_delta = 0.7 * min(rec_res.x, rec_res.y);
    vec2 direction = normalize(p - CENTER);
    vec2 uv_offset_to_closer_p = direction * (interpolate_delta)*vec2(aspect,1.0);
	
	float surface_end_height = get_texture_max(p);
    
    // inspect fragments near mouse	

    float surface_start_height = get_texture_max(p - uv_offset_to_closer_p);

    float viewer_height = get_texture_max(CENTER) + elevation;
    vec2 line_start = vec2(distance(p - uv_offset_to_closer_p, CENTER), surface_start_height);
    vec2 line_end = vec2(distance(p, CENTER), surface_end_height);
    vec2 viewer_loc = vec2(0, viewer_height);
    

    float cross_mag = cross_magnitude(line_start, line_end, viewer_loc);
    return cross_mag;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    #ifdef SHADERTOY
    float elevation = DEVICE_HEIGHT;
    #else
    float elevation = iElevation;
    #endif

    vec2 rec_res = 1.0 / iResolution.xy;
    vec2 p = fragCoord.xy * rec_res;
    {
        float aspect = iResolution.x/iResolution.y;
        float cross_mag = to_surface_face(rec_res, p, aspect, elevation);

        if (-cross_mag <= 0.0){
            fragColor = vec4(0.0, 0.0, 0.0, 1.0);
            return;
        }
    }

    vec2 offset = fragCoord.xy - CENTER_COORD;
    vec2 offset_norm = normalize(offset);
    float offset_dist = length(offset);
    
    float center_height = get_texture_max(vec2(0.5)) + elevation;
    float target_height = get_texture_max(p);

    float target_slope = (target_height - center_height) / offset_dist;
    
    float total = STEP_LENGTH;
    while (total < offset_dist - 0.01 ) {
        vec2 intermediary_coord = (CENTER_COORD + offset_norm * total) * rec_res;
        float intermediary_height = get_texture_max(intermediary_coord);
        
        if ( target_slope < (intermediary_height - center_height) / total){
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