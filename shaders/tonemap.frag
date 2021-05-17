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
