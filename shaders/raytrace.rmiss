/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

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
