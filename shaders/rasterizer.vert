#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable

#include "binding.glsl"
#include "share.glsl"

// clang-format off
layout(set = 0, binding = B_SCENE) uniform _Scene {Scene sceneInfo; } ;
layout(set = 0, binding = B_MATRIX) readonly buffer _Matrix {InstancesMatrices matrices[]; };
// clang-format on

layout(push_constant) uniform _Push
{
  vec3 c_lightDir;
  int c_nbSteps;
  int c_instID;
  int c_matID;
};


// Input
layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_texcoord0;


// Output
layout(location = 0) out vec3 out_worldPos;
layout(location = 1) out vec3 out_normal;
layout(location = 2) out vec2 out_texcoord0;


out gl_PerVertex
{
  vec4 gl_Position;
};


void main()
{
  vec4 worldPos = matrices[c_instID].world * vec4(in_pos, 1.0);
 
  out_normal = vec3(matrices[c_instID].worldIT * vec4(in_normal, 0.0));
  out_worldPos = worldPos.xyz;
  out_texcoord0 = in_texcoord0;

  gl_Position   = sceneInfo.projection * sceneInfo.modelView * worldPos;
}
