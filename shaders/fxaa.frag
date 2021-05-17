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

/**
Basic FXAA implementation based on the code on geeks3d.com with the
modification that the texture2DLod stuff was removed since it's
unsupported by WebGL.

--

From:
https://github.com/mitsuhiko/webgl-meincraft

Copyright (c) 2011 by Armin Ronacher.

Some rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.

    * The names of the contributors may not be used to endorse or
      promote products derived from this software without specific
      prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


// clang-format off
layout(set = 0, binding = 0) uniform sampler2D iChannel0;  // Normal + depth
// clang-format on

layout(location = 0) in vec2 fragCoord;
layout(location = 0) out vec4 fragColor;


#ifndef FXAA_REDUCE_MIN
#define FXAA_REDUCE_MIN (1.0 / 128.0)
#endif
#ifndef FXAA_REDUCE_MUL
#define FXAA_REDUCE_MUL (1.0 / 8.0)
#endif
#ifndef FXAA_SPAN_MAX
#define FXAA_SPAN_MAX 8.0
#endif

// optimized version for mobile, where dependent
// texture reads can be a bottleneck
vec4 fxaa(sampler2D tex,
          vec2      fragCoord,
          vec2      resolution,  //
          vec2      v_rgbNW,
          vec2      v_rgbNE,
          vec2      v_rgbSW,
          vec2      v_rgbSE,
          vec2      v_rgbM)
{
  vec4  color;
  vec2  inverseVP = vec2(1.0 / resolution.x, 1.0 / resolution.y);
  vec3  rgbNW     = texture(tex, v_rgbNW).xyz;
  vec3  rgbNE     = texture(tex, v_rgbNE).xyz;
  vec3  rgbSW     = texture(tex, v_rgbSW).xyz;
  vec3  rgbSE     = texture(tex, v_rgbSE).xyz;
  vec4  texColor  = texture(tex, v_rgbM);
  vec3  rgbM      = texColor.xyz;
  vec3  luma      = vec3(0.299, 0.587, 0.114);
  float lumaNW    = dot(rgbNW, luma);
  float lumaNE    = dot(rgbNE, luma);
  float lumaSW    = dot(rgbSW, luma);
  float lumaSE    = dot(rgbSE, luma);
  float lumaM     = dot(rgbM, luma);
  float lumaMin   = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
  float lumaMax   = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

  vec2 dir;
  dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
  dir.y = ((lumaNW + lumaSW) - (lumaNE + lumaSE));

  float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * (0.25 * FXAA_REDUCE_MUL), FXAA_REDUCE_MIN);

  float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
  dir = min(vec2(FXAA_SPAN_MAX, FXAA_SPAN_MAX), max(vec2(-FXAA_SPAN_MAX, -FXAA_SPAN_MAX), dir * rcpDirMin)) * inverseVP;

  vec3 rgbA = 0.5
              * (texture(tex, fragCoord * inverseVP + dir * (1.0 / 3.0 - 0.5)).xyz
                 + texture(tex, fragCoord * inverseVP + dir * (2.0 / 3.0 - 0.5)).xyz);
  vec3 rgbB =
      rgbA * 0.5
      + 0.25 * (texture(tex, fragCoord * inverseVP + dir * -0.5).xyz + texture(tex, fragCoord * inverseVP + dir * 0.5).xyz);

  float lumaB = dot(rgbB, luma);
  if((lumaB < lumaMin) || (lumaB > lumaMax))
    color = vec4(rgbA, texColor.a);
  else
    color = vec4(rgbB, texColor.a);
  return color;
}

void main()
{
  vec2 resolution = vec2(textureSize(iChannel0, 0));
  vec2 fragCoord  = gl_FragCoord.xy;

  vec2 inverseVP = 1.0 / resolution.xy;
  vec2 v_rgbNW   = (fragCoord + vec2(-1.0, -1.0)) * inverseVP;
  vec2 v_rgbNE   = (fragCoord + vec2(1.0, -1.0)) * inverseVP;
  vec2 v_rgbSW   = (fragCoord + vec2(-1.0, 1.0)) * inverseVP;
  vec2 v_rgbSE   = (fragCoord + vec2(1.0, 1.0)) * inverseVP;
  vec2 v_rgbM    = vec2(fragCoord * inverseVP);


  vec4 result = fxaa(iChannel0, fragCoord, resolution,  //
                     v_rgbNW, v_rgbNE, v_rgbSW, v_rgbSE, v_rgbM);

  fragColor = result;
}
