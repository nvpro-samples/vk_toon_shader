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

#pragma once

#include <vulkan/vulkan.hpp>

// Choosing the allocator to use
// #define ALLOC_DMA
 #define ALLOC_DEDICATED
// #define ALLOC_VMA
#include <nvvk/resourceallocator_vk.hpp>

#if defined(ALLOC_DMA)
#include <nvvk/memallocator_dma_vk.hpp>
typedef nvvk::ResourceAllocatorDma Allocator;
#elif defined(ALLOC_VMA)
#include <nvvk/memallocator_vma_vk.hpp>
typedef nvvk::ResourceAllocatorVma Allocator;
#else
typedef nvvk::ResourceAllocatorDedicated Allocator;
#endif

using vkDT = vk::DescriptorType;
using vkDS = vk::DescriptorSetLayoutBinding;
using vkSS = vk::ShaderStageFlagBits;
using vkCB = vk::CommandBufferUsageFlagBits;
using vkBU = vk::BufferUsageFlagBits;
using vkIU = vk::ImageUsageFlagBits;
using vkMP = vk::MemoryPropertyFlagBits;


// Utility to time the execution of something resetting the timer
// on each elapse call
// Usage:
// {
//   MilliTimer timer;
//   ... stuff ...
//   double time_elapse = timer.elapse();
// }
#include "nvmath/nvmath.h"
#include <chrono>

struct MilliTimer
{
  MilliTimer() { reset(); }
  void   reset() { startTime = std::chrono::high_resolution_clock::now(); }
  double elapse()
  {
    auto now  = std::chrono::high_resolution_clock::now();
    auto t    = std::chrono::duration_cast<std::chrono::microseconds>(now - startTime).count() / 1000.0;
    startTime = now;
    return t;
  }
  std::chrono::high_resolution_clock::time_point startTime;
};

inline vk::ClearColorValue makeClearColor(nvmath::vec3f& _c)
{
  vk::ClearColorValue cc;
  cc.float32[0] = _c.x;
  cc.float32[1] = _c.y;
  cc.float32[2] = _c.z;
  cc.float32[3] = 0;
  return cc;
}
