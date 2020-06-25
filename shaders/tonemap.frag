#version 450
#extension GL_GOOGLE_include_directive : enable

//----------------------------------------------------------------------------
// Use for tonemapping the incoming image (full screen quad)
//


// Tonemapping functions
#include "tonemapping.glsl"

layout(location = 0) in vec2 outUV;
layout(location = 0) out vec4 fragColor;

layout(set = 0, binding = 0) uniform sampler2D inImage;

layout(push_constant) uniform shaderInformation
{
  int   tonemapper;  // tonemapper to use
  float gamma;       // Default 2.2
  float exposure;    // Overal exposure
}
pushc;

void main()
{
  vec2 uv    = outUV;
  vec4 color = texture(inImage, uv);

  fragColor = vec4(toneMap(color.rgb, pushc.tonemapper, pushc.gamma, pushc.exposure), color.a);
}
