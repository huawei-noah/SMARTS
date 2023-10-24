// Visibility check

// -----------------------------------------------
#define STEP_LENGTH 0.2
#define CENTER_COORD iResolution.xy * 0.5


void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 rec_res = 1.0 / iResolution.xy;
    vec2 p = fragCoord.xy * rec_res;

    vec2 offset = fragCoord.xy - CENTER_COORD;
    vec2 offset_norm = normalize(offset);
    float offset_dist = length(offset);
    
    if (texture(iChannel1, vec2(0.5)).x <= 0.0){
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    
    float center_height = texture(iChannel0, vec2(0.5)).x * TOPOLOGY_SCALING_FACTOR + DEVICE_HEIGHT;
    float target_height = texture(iChannel0, p).x * TOPOLOGY_SCALING_FACTOR;

    float target_slope = (target_height - center_height) / offset_dist;
    
    float total = STEP_LENGTH;
    while (total < offset_dist - 0.01 ) {
        vec2 intermediary_coord = (CENTER_COORD + offset_norm * total) * rec_res;
        float intermediary_height = texture(iChannel0, intermediary_coord).x * TOPOLOGY_SCALING_FACTOR;
        
        if ( target_slope < ((intermediary_height - center_height) * TOPOLOGY_SCALING_FACTOR ) / total){
            fragColor = vec4(0.0, 0.0, 0.0, 1.0);
            return;
        }
        
        
        total += STEP_LENGTH;
    }
    

    fragColor = vec4(1.0,1.0,1.0,1.0);
}