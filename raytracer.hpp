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

//////////////////////////////////////////////////////////////////////////
// Raytracing implementation
//
// There are 2 descriptor sets
// (0) - Acceleration structure and result image
// (1) - Various buffers: vertices, indices, matrices, Material and Textures
//
//////////////////////////////////////////////////////////////////////////

#include <vulkan/vulkan.hpp>

#include "vk_util.hpp"

#include "nvvk/commands_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/raytraceNV_vk.hpp"

// Structure used for retrieving the primitive information in the closest hit
// The gl_InstanceCustomIndexNV
struct RtPrimitiveLookup
{
  uint32_t indexOffset;
  uint32_t vertexOffset;
  int      materialIndex;
};

class Raytracer
{
public:
  Raytracer();

  // Initializing the allocator and querying the raytracing properties
  void setup(const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t queueIndex, nvvkpp::ResourceAllocator* allocator);

  bool isValid() { return m_bValid; }

  // Return the rendered image
  const std::vector<nvvk::Texture>& outputImages() const;
  int                               maxFrames() const;

  void destroy();

  // Creating the two images where the result is stored
  void createOutputImages(vk::Extent2D size);

  // Create a descriptor set holding the acceleration structure and the output image
  void createDescriptorSet();

  // Will be called when resizing the window
  void updateDescriptorSet();

  // Pipeline with all shaders, including the 3 descriptor layouts.
  void createPipeline(const vk::DescriptorSetLayout& sceneDescSetLayout);

  // The SBT, storing in a buffer the calling handles of each shader group
  void createShadingBindingTable();

  // Executing the raytracing
  void run(const vk::CommandBuffer& cmdBuf, const vk::DescriptorSet& sceneDescSet, int frame = 0);

  // To control the raytracer
  bool uiSetup();

  nvvkpp::RaytracingBuilderNV& builder() { return m_rtBuilder; }

  void setPrimitiveLookup(const std::vector<RtPrimitiveLookup>& primitiveLookup)
  {
    nvvkpp::ScopeCommandBuffer cmdBuf(m_device, m_queueIndex);
    m_rtPrimLookup = m_alloc->createBuffer(cmdBuf, primitiveLookup, vk::BufferUsageFlagBits::eStorageBuffer);
    m_debug.setObjectName(m_rtPrimLookup.buffer, "PrimitiveInfo");
  }

  void setClearColor(nvmath::vec3f& _color);
  void setToonSteps(int nbStep);
  void setToonLightDir(nvmath::vec3f lightDir);

private:
  struct PushConstant
  {
    nvmath::vec3f backgroundColor{1, 1, 1};
    int           frame{0};  // Current frame number
    nvmath::vec3f lightDir{-1, -1, -1};
    float         maxRayLenght{100000};
    int           samples{1};  // samples per frame
    int           nbSteps{3};  // Dither
  } m_pushC;

  int m_maxFrames{50};  // Max iterations

  vk::PhysicalDeviceRayTracingPropertiesNV m_rtProperties;
  std::vector<nvvk::Texture>               m_raytracingOutput;  // many outputs

  // Raytracer
  nvvk::Buffer                                       m_sbtBuffer;
  nvvkpp::RaytracingBuilderNV                       m_rtBuilder;
  nvvkpp::DescriptorSetBindings                     m_descSetLayoutBind;
  vk::DescriptorPool                                 m_descPool;
  vk::DescriptorSetLayout                            m_descSetLayout;
  vk::DescriptorSet                                  m_descSet;
  vk::PipelineLayout                                 m_pipelineLayout;
  vk::Pipeline                                       m_pipeline;
  vk::Extent2D                                       m_outputSize;
  nvvkpp::DescriptorSetBindings                     m_binding;
  nvvk::Buffer                                       m_rtPrimLookup;
  std::vector<vk::RayTracingShaderGroupCreateInfoNV> m_groups;


  // Vulkan
  bool                        m_bValid{false};
  vk::Device                  m_device;
  nvvk::DebugUtil             m_debug;
  uint32_t                    m_queueIndex;
  nvvkpp::ResourceAllocator* m_alloc{nullptr};
};
