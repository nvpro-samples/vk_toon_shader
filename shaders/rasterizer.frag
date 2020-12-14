#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable

#include "binding.glsl"
#include "share.glsl"

// clang-format off
layout(set = 0, binding = B_SCENE) uniform _Scene {Scene sceneInfo; };
layout(set = 0, binding = B_MATERIAL) readonly buffer _Material {GltfShadeMaterial materials[];};
layout(set = 0, binding = B_TEXTURES) uniform sampler2D texturesMap[]; // all textures
// clang-format on

layout(push_constant) uniform _Push
{
  vec3 c_lightDir;
  int  c_nbSteps;
  int  c_instID;
  int  c_matID;
};

// Incoming
layout(location = 0) in vec3 in_worldPos;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_texcoord0;

// Outgoing
layout(location = 0) out vec4 out_color;
layout(location = 1) out vec4 out_bufferEncoded;  // xy: normal, z, object ID


// Debug utility
vec3 integerToColor(int val)
{
  const vec3 freq = vec3(1.33333f, 2.33333f, 3.33333f);
  return vec3(sin(freq * val) * .5 + .5);
}

void main()
{
  // Retrieve the material on this hit
  GltfShadeMaterial material = materials[c_matID];

  // Albedo
  vec3 baseColor = material.pbrBaseColorFactor.rgb;
  if(material.pbrBaseColorTexture > -1)
    baseColor *= texture(texturesMap[nonuniformEXT(material.pbrBaseColorTexture)], in_texcoord0).rgb;

  // Color Result
  vec3  toLight   = normalize(-c_lightDir);
  float intensity = toonShading(toLight, in_normal, c_nbSteps);

  out_color.xyz = max(0.1f, intensity) * baseColor.xyz;  // keeping ambient
  out_color.a   = 1;

  // Encoding data buffer
  vec2  encodedNormal = encode(normalize(in_normal));
  float depth         = length(in_worldPos - sceneInfo.camPos.xyz);
  float fInstanceID   = intBitsToFloat(c_instID + 1);
  out_bufferEncoded   = vec4(encodedNormal, depth, fInstanceID);
}
