#version 330 core
// Surface direction check
#define DEVICE_HEIGHT 10
#define TOPOLOGY_SCALING_FACTOR 1.0
#define CENTER vec2(0.5)
//#define SHADERTOY
// -----------------------------------------------

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

float get_texture_max( vec2 uv ) {
   return texture(iChannel0, uv).x + texture(iChannel1, uv).x;
}
#endif

float cross_magnitude(vec2 lp1, vec2 lp2, vec2 p1){
    vec2 v1 = lp2 - lp1;
    vec2 v2 = lp2 - p1;
    
    return v1.x * v2.y - v1.y * v2.x;
}

float to_surface_face(vec2 rec_res, vec2 p, float aspect) {
    // Determine the minimum delta needed to determine the facing direction of the surface
    float interpolate_delta = 0.7 * min(rec_res.x, rec_res.y);
    vec2 direction = normalize(p - CENTER);
    vec2 uv_offset_to_closer_p = direction * (interpolate_delta)*vec2(aspect,1.0);
	
	float surface_end_height = get_texture_max(p) * TOPOLOGY_SCALING_FACTOR;
    
    // inspect fragments near mouse	

    float surface_start_height = get_texture_max(p - uv_offset_to_closer_p) * TOPOLOGY_SCALING_FACTOR;

    float viewer_height = get_texture_max(CENTER) * TOPOLOGY_SCALING_FACTOR + DEVICE_HEIGHT;
    vec2 line_start = vec2(distance(p - uv_offset_to_closer_p, CENTER), surface_start_height);
    vec2 line_end = vec2(distance(p, CENTER), surface_end_height);
    vec2 viewer_loc = vec2(0, viewer_height);
    

    float cross_mag = cross_magnitude(line_start, line_end, viewer_loc);
    return cross_mag;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 rec_res = 1.0 / iResolution.xy;
    vec2 p = fragCoord.xy * rec_res;
    float aspect = iResolution.x/iResolution.y;

    float cross_mag = to_surface_face(rec_res, p, aspect);
    fragColor = vec4(vec3(-cross_mag), 1.0);
}

#ifndef SHADERTOY
void main(){
    mainImage(p3d_Color, gl_FragCoord.xy);
}
#endif