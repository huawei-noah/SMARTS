// The MIT License
// Copyright © 2013 Inigo Quilez
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// https://www.youtube.com/c/InigoQuilez
// https://iquilezles.org


// Simplex Noise (http://en.wikipedia.org/wiki/Simplex_noise), a type of gradient noise
// that uses N+1 vertices for random gradient interpolation instead of 2^N as in regular
// latice based Gradient Noise.

// All noise functions here:
//
// https://www.shadertoy.com/playlist/fXlXzf&from=0&num=12
//#define SEE_DIRECTION // See the direction to the pixel
#define SHADERTOY
#define CENTER 0.5
#define HEIGHT 1.0
#define RECIPROCAL_BYTE 0.0039215686274509803921568627451

vec2 hash( vec2 p ) // replace this by something better
{
	p = vec2( dot(p,vec2(127.1,311.7)), dot(p,vec2(269.5,183.3)) );
	return -1.0 + 2.0*fract(sin(p)*43758.5453123);
}

float noise( in vec2 p )
{
    const float K1 = 0.366025404; // (sqrt(3)-1)/2;
    const float K2 = 0.211324865; // (3-sqrt(3))/6;

	vec2  i = floor( p + (p.x+p.y)*K1 );
    vec2  a = p - i + (i.x+i.y)*K2;
    float m = step(a.y,a.x); 
    vec2  o = vec2(m,1.0-m);
    vec2  b = a - o + K2;
	vec2  c = a - 1.0 + 2.0*K2;
    vec3  h = max( 0.5-vec3(dot(a,a), dot(b,b), dot(c,c) ), 0.0 );
	vec3  n = h*h*h*h*vec3( dot(a,hash(i+0.0)), dot(b,hash(i+o)), dot(c,hash(i+1.0)));
    return dot( n, vec3(70.0) );
}

float noise_with_octaves( in vec2 uv, in mat2 m ){
	float f = 0.0;
    uv *= 5.0;

    f  = 0.5000*noise( uv ); uv = m*uv;
    f += 0.2500*noise( uv ); uv = m*uv;
    f += 0.1250*noise( uv ); uv = m*uv;
    f += 0.0625*noise( uv ); uv = m*uv;
    return f;
}

// -----------------------------------------------

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 rec_res = 1.0 / iResolution.xy;
    vec2 p = fragCoord.xy * rec_res;
    float aspect = iResolution.x/iResolution.y;

    #ifdef SHADERTOY
	vec2 uv = p*vec2(aspect,1.0) + vec2(iTime * 0.1);
    #else
    vec2 uv = p*vec2(aspect,1.0);
    #endif

    
    // Determine the minimum delta needed to determine the facing direction of the surface
    float interpolate_delta = 1.0 * min(rec_res.x, rec_res.y);
    vec2 direction = normalize(p - vec2(0.5));
    vec2 uv_offset_to_closer_p = direction * (interpolate_delta)*vec2(aspect,1.0);
    
    #ifdef SEE_DIRECTION
    uv_offset_to_closer_p *= 1000.0;
    fragColor = vec4(uv_offset_to_closer_p, 0, 1.0);
    return;
    #endif
	
	float f = 0.0;
    float x, y, z;
    mat2 m = mat2( 1.6,  1.2, -1.2,  1.6 );

    f = noise_with_octaves(uv, m);
    // inspect fragments near mouse	
	if( distance(fragCoord, iMouse.xy) < 30.0 )
	{
		f = 0.5 + 0.5*f;
        f *= 0.3 + f;
        fragColor = vec4( f, f, f, 1.0 );
	}
    else {
        float o = noise_with_octaves(uv - uv_offset_to_closer_p, m);
        vec3 c = cross(
            //vec3(direction.y, -direction.x, 0),
            vec3(1, 0, 0),
            normalize(vec3(interpolate_delta, 0, (f - o) * RECIPROCAL_BYTE))
        );
        fragColor = vec4(c.yyy * 25.5, 1.0);
    }


    if( distance(fragCoord, iResolution.xy/2.0) < 4.0 ) {
        fragColor += vec4(0.4, 0.0, 0.0, 1.0);
    }
}