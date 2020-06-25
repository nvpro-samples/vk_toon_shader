#version 460
#extension GL_NV_ray_tracing : require

layout(location = 1) rayPayloadInNV bool payload_isHit;

//-------------------------------------------------------------------------------------------------
// This will be executed when sending shadow rays and missing all geometries
// - There are no hit shader for the shadow ray, therefore
// - Before calling Trace, set payload_isHit=true
// - The default anyhit, closesthit won't change isShadowed, but if nothing is hit, it will be
//   set to false.
//-------------------------------------------------------------------------------------------------

void main()
{
  payload_isHit = false;
}
