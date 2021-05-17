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

 // by Jan Eric Kyprianidis <www.kyprianidis.com>
#version 450

layout(set = 0, binding = 0) uniform sampler2D src;
layout(location = 0) out vec4 fragColor;
layout(location = 0) in vec2 fragCoord;

void main(void)
{
  vec2 uv = gl_FragCoord.xy / vec2(textureSize(src, 0));
  vec3 g  = texture(src, uv).xyz;

  float lambda1 = 0.5 * (g.y + g.x + sqrt(g.y * g.y - 2.0 * g.x * g.y + g.x * g.x + 4.0 * g.z * g.z));
  float lambda2 = 0.5 * (g.y + g.x - sqrt(g.y * g.y - 2.0 * g.x * g.y + g.x * g.x + 4.0 * g.z * g.z));

  vec2 v = vec2(lambda1 - g.x, -g.z);
  vec2 t;
  if(length(v) > 0.0)
  {
    t = normalize(v);
  }
  else
  {
    t = vec2(0.0, 1.0);
  }

  float phi = atan(t.y, t.x);

  float A = (lambda1 + lambda2 > 0.0) ? (lambda1 - lambda2) / (lambda1 + lambda2) : 0.0;

  fragColor = vec4(t, phi, A);
}
