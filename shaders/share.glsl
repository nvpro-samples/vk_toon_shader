
//
// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//


#include "gltf.glsl"

const highp float M_PI   = 3.14159265358979323846f;  // pi
const highp float M_PI_2 = 1.57079632679489661923f;  // pi/2
const highp float M_PI_4 = 0.785398163397448309616;  // pi/4
const highp float M_1_PI = 0.318309886183790671538;  // 1/pi
const highp float M_2_PI = 0.636619772367581343076;  // 2/pi

//-------------------------------------------------------------------------------------------------
// random number generator based on the Optix SDK
//-------------------------------------------------------------------------------------------------

uint tea(uint val0, uint val1)
{
  uint v0 = val0;
  uint v1 = val1;
  uint s0 = 0u;

  for(uint n = 0u; n < 16u; n++)
  {
    s0 += 0x9e3779b9u;
    v0 += ((v1 << 4u) + 0xa341316cu) ^ (v1 + s0) ^ ((v1 >> 5u) + 0xc8013ea4u);
    v1 += ((v0 << 4u) + 0xad90777du) ^ (v0 + s0) ^ ((v0 >> 5u) + 0x7e95761eu);
  }

  return v0;
}

// Generate random unsigned int in [0, 2^24)
uint lcg(inout uint prev)
{
  uint LCG_A = 1664525u;
  uint LCG_C = 1013904223u;
  prev       = (LCG_A * prev + LCG_C);
  return prev & 0x00FFFFFF;
}

uint lcg2(inout uint prev)
{
  prev = (prev * 8121 + 28411) % 134456;
  return prev;
}

// Generate random float in [0, 1)
float rnd(inout uint prev)
{
  return (float(lcg(prev)) / float(0x01000000));
}


vec2 rnd2(inout uint prev)
{
  return vec2(rnd(prev), rnd(prev));
}


//-------------------------------------------------------------------------------------------------
// Avoiding self intersections (see Ray Tracing Gems, Ch. 6)
//-------------------------------------------------------------------------------------------------

vec3 offsetRay(in vec3 p, in vec3 n)
{
  const float intScale   = 256.0f;
  const float floatScale = 1.0f / 65536.0f;
  const float origin     = 1.0f / 32.0f;

  ivec3 of_i = ivec3(intScale * n.x, intScale * n.y, intScale * n.z);

  vec3 p_i = vec3(intBitsToFloat(floatBitsToInt(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
                  intBitsToFloat(floatBitsToInt(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
                  intBitsToFloat(floatBitsToInt(p.z) + ((p.z < 0) ? -of_i.z : of_i.z)));

  return vec3(abs(p.x) < origin ? p.x + floatScale * n.x : p_i.x,  //
              abs(p.y) < origin ? p.y + floatScale * n.y : p_i.y,  //
              abs(p.z) < origin ? p.z + floatScale * n.z : p_i.z);
}


//-------------------------------------------------------------------------------------------------
// Toon Shading : apply quatization on lambertian when nbSteps > 0
//-------------------------------------------------------------------------------------------------

float toonShading(in vec3  L,       // Vector to light
                  in vec3  N,       // Normal
                  in float nbSteps  // Quatization steps
)
{
  float NdotL = dot(N, L);
  if(nbSteps > 0.f)
  {
    const float inv = 1.f / nbSteps;
    NdotL           = floor(NdotL * (1.f + inv * 0.5f) * nbSteps) * inv;
  }
  return NdotL;
}

//-------------------------------------------------------------------------------------------------
// Ray tracer Payloads
//-------------------------------------------------------------------------------------------------

// Payload for raytracing
struct PerRayData_raytrace
{
  vec4  result;
  vec3  importance;
  float roughness;
  uint  seed;
  int   rayDepth;
  vec3  normal;
  float depth;
  int   objId;
};

// Payload for ray picking
struct PerRayData_pick
{
  vec4 worldPos;
  vec4 barycentrics;
  uint instanceID;
  uint instanceCustomID;
  uint primitiveID;
};


//-------------------------------------------------------------------------------------------------
// Structures in buffers
//-------------------------------------------------------------------------------------------------

// Primitive Mesh information, use for finding geometry data
struct PrimMeshInfo
{
  uint indexOffset;
  uint vertexOffset;
  int  materialIndex;
};

// Matrices buffer for all instances
struct InstancesMatrices
{
  mat4 world;
  mat4 worldIT;
};

// Scene information
struct Scene
{
  mat4 projection;
  mat4 modelView;
  vec4 camPos;
};


//-------------------------------------------------------------------------------------------------
// sRGB to linear approximation
// see http://chilliant.blogspot.com/2012/08/srgb-approximations-for-hlsl.html
//-------------------------------------------------------------------------------------------------
vec4 SRGBtoLINEAR(vec4 srgbIn, float gamma)
{
  return vec4(pow(srgbIn.xyz, vec3(gamma)), srgbIn.w);
}

//-------------------------------------------------------------------------------------------------
// Normal Encoding: Lambert azimuthal equal-area projection
// see Method #4 https://aras-p.info/texts/CompactNormalStorage.html
//-------------------------------------------------------------------------------------------------
vec2 encode(vec3 n)
{
  float p = sqrt(n.z * 8 + 8);
  return vec2(n.xy / p + 0.5);
}
// Normal Decoding
vec3 decode(vec2 enc)
{
  vec2  fenc = enc * 4 - 2;
  float f    = dot(fenc, fenc);
  float g    = sqrt(1 - f / 4);
  vec3  n;
  n.xy = fenc * g;
  n.z  = 1 - f / 2;
  return n;
}
