#version 330 core
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


// -----------------------------------------------

//#define SHADERTOY

#define DENSITY_U 1.6 * 0.1
#define DENSITY_V 1.2 * 0.1

#ifdef SHADERTOY
#define scale 1.0

#else
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

// Output color
out vec4 p3d_Color;

// inputs
uniform vec2 iResolution;
uniform float iHeading;
uniform vec2 iTranslation;
uniform float scale;

uniform sampler2D iChannel0;
#endif

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = ( fragCoord.xy - .5 * iResolution.xy ) / iResolution.y;

    #ifdef SHADERTOY
    float rotation = iTime;
    vec2 translation = iMouse.xy;
    #else
    float rotation = -iHeading; // Turn opposite the rotation of the object
    vec2 translation = iTranslation * scale;
    #endif

    float s = sin(rotation);
    float c = cos(rotation);
    uv = uv * mat2(c, s, -s, c) + translation / iResolution.y;

	float f = 0.0;
    float inv_s = 1.0 / scale;
    mat2 m = mat2( DENSITY_U,  DENSITY_V, -DENSITY_V,  DENSITY_U ) * inv_s;
    f = noise_with_octaves(uv, m);

    vec2 p = fragCoord.xy * 1.0 / iResolution.xy;
    if (texture(iChannel0, p).r < .2) {
        f = f + 2.0 * abs(noise_with_octaves(uv, m * 2.0));
    }
    //f = 0.5 + 0.5*f;
    //f *= 0.3 + f;

    fragColor = vec4( f, f, f, f );
}

#ifndef SHADERTOY
void main(){
    mainImage(p3d_Color, gl_FragCoord.xy);
}
#endif