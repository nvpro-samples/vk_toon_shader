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
  vec2 src_size = vec2(textureSize(src, 0));
  vec2 uv       = gl_FragCoord.xy / src_size;
  vec2 d        = 1.0 / src_size;

  vec3 c = texture(src, uv).xyz;
  vec3 u = (-1.0 * texture(src, uv + vec2(-d.x, -d.y)).xyz + -2.0 * texture(src, uv + vec2(-d.x, 0.0)).xyz
            + -1.0 * texture(src, uv + vec2(-d.x, d.y)).xyz + +1.0 * texture(src, uv + vec2(d.x, -d.y)).xyz
            + +2.0 * texture(src, uv + vec2(d.x, 0.0)).xyz + +1.0 * texture(src, uv + vec2(d.x, d.y)).xyz)
           / 4.0;

  vec3 v = (-1.0 * texture(src, uv + vec2(-d.x, -d.y)).xyz + -2.0 * texture(src, uv + vec2(0.0, -d.y)).xyz
            + -1.0 * texture(src, uv + vec2(d.x, -d.y)).xyz + +1.0 * texture(src, uv + vec2(-d.x, d.y)).xyz
            + +2.0 * texture(src, uv + vec2(0.0, d.y)).xyz + +1.0 * texture(src, uv + vec2(d.x, d.y)).xyz)
           / 4.0;

  fragColor = vec4(dot(u, u), dot(v, v), dot(u, v), 1.0);
}
