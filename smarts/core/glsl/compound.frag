#define SPOT_RADIUS 100.0

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 rec_res = 1.0 / iResolution.xy;
    vec2 p = fragCoord.xy * rec_res;

    fragColor = texture(iChannel2, p);
    
    
    #ifdef SPOT_CHECK
    if( distance(fragCoord, iMouse.xy) < 30.0 )
	{
        fragColor = texture(iChannel2, p);
    }
	else if( distance(fragCoord, iMouse.xy) < SPOT_RADIUS )
	{
        float aspect = iResolution.x/iResolution.y;
        vec2 uv = p*vec2(aspect,1.0);
        mat2 m = mat2( DENSITY_U,  DENSITY_V, -DENSITY_V,  DENSITY_U );
        float f = 0.0;
        f = noise_with_octaves(uv, m);
		f = 0.5 + 0.5*f;
        f *= 0.3 + f;
        fragColor = vec4( f, f, f, 1.0 );
	}
    #endif
}