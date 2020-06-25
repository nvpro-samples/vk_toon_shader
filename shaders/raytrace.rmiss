#version 460
#extension GL_NV_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable

//-------------------------------------------------------------------------------------------------
// Default miss shader for the raytracer
// - Return the background color
//-------------------------------------------------------------------------------------------------

#include "binding.glsl"
#include "share.glsl"

layout(location = 0) rayPayloadInNV PerRayData_raytrace payload;

layout(push_constant) uniform _Push
{
  vec3  c_backgroundColor;  // Miss color
  int   c_frame;            // Current frame
  vec3  c_lightDir;         //
  float c_maxRayLenght;     // Trace depth
  int   c_samples;          // Number of samples per pixel
  int   c_nbSteps;          // Toon shading steps
};

void main()
{
  payload.result = vec4(c_backgroundColor, 0);
  payload.depth  = -1;  // Will stop rendering
}
