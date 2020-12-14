#version 460
#extension GL_NV_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable

#include "binding.glsl"
#include "share.glsl"


layout(location = 0) rayPayloadInNV PerRayData_raytrace prd;  // incoming from raygen
layout(location = 1) rayPayloadNV bool payload_isHit;         // shadow

layout(push_constant) uniform _Push
{
  vec3  c_backgroundColor;
  int   c_frame;  // Current frame
  vec3  c_lightDir;
  float c_maxRayLenght;  // Trace depth
  int   c_samples;       // Number of samples per pixel
  int   c_nbSteps;
};


// Raytracing hit attributes: barycentrics
hitAttributeNV vec2 attribs;

// clang-format off
layout(set = 0, binding = 0) uniform accelerationStructureNV topLevelAS;
layout(set = 0, binding = 2) readonly buffer _InstanceInfo {PrimMeshInfo primInfo[];};

layout(set = 1, binding = B_VERTICES) readonly buffer _VertexBuf {float vertices[];};
layout(set = 1, binding = B_INDICES) readonly buffer _Indices {uint indices[];};
layout(set = 1, binding = B_NORMALS) readonly buffer _NormalBuf {float normals[];};
layout(set = 1, binding = B_TEXCOORDS) readonly buffer _TexCoordBuf {float texcoord0[];};
layout(set = 1, binding = B_MATERIAL) readonly buffer _MaterialBuffer {GltfShadeMaterial materials[];};
layout(set = 1, binding = B_TEXTURES) uniform sampler2D texturesMap[]; // all textures
// clang-format on


// Return the vertex position
vec3 getVertex(uint index)
{
  vec3 vp;
  vp.x = vertices[3 * index + 0];
  vp.y = vertices[3 * index + 1];
  vp.z = vertices[3 * index + 2];
  return vp;
}

vec3 getNormal(uint index)
{
  vec3 vp;
  vp.x = normals[3 * index + 0];
  vp.y = normals[3 * index + 1];
  vp.z = normals[3 * index + 2];
  return vp;
}

vec2 getTexCoord(uint index)
{
  vec2 vp;
  vp.x = texcoord0[2 * index + 0];
  vp.y = texcoord0[2 * index + 1];
  return vp;
}

// Structure of what a vertex is
struct ShadingState
{
  vec3 pos;
  vec3 normal;
  vec3 geom_normal;
  vec2 texcoord0;
  uint matIndex;
};

//--------------------------------------------------------------
// Getting the interpolated vertex
// gl_InstanceID gives the Instance Info
// gl_PrimitiveID gives the triangle for this instance
//
ShadingState getShadingState()
{
  // Retrieve the Primitive mesh buffer information
  PrimMeshInfo pinfo = primInfo[gl_InstanceCustomIndexNV];

  // Getting the 'first index' for this mesh (offset of the mesh + offset of the triangle)
  uint indexOffset  = pinfo.indexOffset + (3 * gl_PrimitiveID);
  uint vertexOffset = pinfo.vertexOffset;           // Vertex offset as defined in glTF
  uint matIndex     = max(0, pinfo.materialIndex);  // material of primitive mesh

  // Getting the 3 indices of the triangle (local)
  ivec3 triangleIndex = ivec3(indices[indexOffset + 0], indices[indexOffset + 1], indices[indexOffset + 2]);
  triangleIndex += ivec3(vertexOffset);  // (global)

  const vec3 barycentric = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

  // Position
  const vec3 pos0           = getVertex(triangleIndex.x);
  const vec3 pos1           = getVertex(triangleIndex.y);
  const vec3 pos2           = getVertex(triangleIndex.z);
  const vec3 position       = pos0 * barycentric.x + pos1 * barycentric.y + pos2 * barycentric.z;
  const vec3 world_position = vec3(gl_ObjectToWorldNV * vec4(position, 1.0));

  // Normal
  const vec3 nrm0         = getNormal(triangleIndex.x);
  const vec3 nrm1         = getNormal(triangleIndex.y);
  const vec3 nrm2         = getNormal(triangleIndex.z);
  vec3       normal       = normalize(nrm0 * barycentric.x + nrm1 * barycentric.y + nrm2 * barycentric.z);
  const vec3 world_normal = normalize(vec3(normal * gl_WorldToObjectNV));
  const vec3 geom_normal  = normalize(cross(pos1 - pos0, pos2 - pos0));

  // Move normal to same side as geometric normal
  if(dot(normal, geom_normal) <= 0)
  {
    normal *= -1.0f;
  }

  // Texture coord
  const vec2 uv0       = getTexCoord(triangleIndex.x);
  const vec2 uv1       = getTexCoord(triangleIndex.y);
  const vec2 uv2       = getTexCoord(triangleIndex.z);
  const vec2 texcoord0 = uv0 * barycentric.x + uv1 * barycentric.y + uv2 * barycentric.z;

  // Final shading state
  ShadingState state;
  state.pos         = world_position;
  state.normal      = world_normal;
  state.geom_normal = geom_normal;
  state.texcoord0   = texcoord0;
  state.matIndex    = matIndex;

  return state;
}


void main()
{
  // Get the shading information
  ShadingState state = getShadingState();  //ind, vertexOffset, barycentrics);

  // cast a shadow ray; assuming light is always outside
  vec3 origin  = state.pos;
  vec3 toLight = normalize(-c_lightDir);

  payload_isHit = true;  // Assuming the ray has hit something
  uint rayFlags = gl_RayFlagsTerminateOnFirstHitNV | gl_RayFlagsSkipClosestHitShaderNV;
  traceNV(topLevelAS,      // acceleration structure
          rayFlags,        // rayFlags
          0xFF,            // cullMask
          0,               // sbtRecordOffset
          0,               // sbtRecordStride
          1,               // missIndex
          origin,          // ray origin
          0.01,            // ray min range
          toLight,         // ray direction
          c_maxRayLenght,  // ray max range
          1                // payload layout(location = 1)
  );

  // Retrieve the material on this hit
  GltfShadeMaterial material = materials[state.matIndex];

  // The albedo may be defined from a base texture or a flat color
  vec3 baseColor = material.pbrBaseColorFactor.rgb;
  if(material.pbrBaseColorTexture > -1)
    baseColor *= texture(texturesMap[nonuniformEXT(material.pbrBaseColorTexture)], state.texcoord0).rgb;

  // If isHit is ture, means the ray hit an object, therefore cannot see the light and it is under shadow
  float intensity;
  if(payload_isHit)
    intensity = 0.0f;
  else
    intensity = toonShading(toLight, state.normal, c_nbSteps);

  // Result color
  prd.result.xyz = max(0.1f, intensity) * baseColor.xyz;  // keeping ambient
  prd.result.a   = 1;

  // Result data
  prd.normal = state.normal;
  prd.depth  = gl_HitTNV;
  prd.objId  = gl_InstanceID + 1;
}
