/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
  void setup(const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t queueIndex, nvvk::Allocator* allocator);

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

  nvvk::RaytracingBuilderNV& builder() { return m_rtBuilder; }

  void setPrimitiveLookup(const std::vector<RtPrimitiveLookup>& primitiveLookup)
  {
    nvvk::ScopeCommandBuffer cmdBuf(m_device, m_queueIndex);
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
  nvvk::RaytracingBuilderNV                          m_rtBuilder;
  nvvk::DescriptorSetBindings                        m_descSetLayoutBind;
  vk::DescriptorPool                                 m_descPool;
  vk::DescriptorSetLayout                            m_descSetLayout;
  vk::DescriptorSet                                  m_descSet;
  vk::PipelineLayout                                 m_pipelineLayout;
  vk::Pipeline                                       m_pipeline;
  vk::Extent2D                                       m_outputSize;
  nvvk::DescriptorSetBindings                        m_binding;
  nvvk::Buffer                                       m_rtPrimLookup;
  std::vector<vk::RayTracingShaderGroupCreateInfoNV> m_groups;


  // Vulkan
  bool             m_bValid{false};
  vk::Device       m_device;
  nvvk::DebugUtil  m_debug;
  uint32_t         m_queueIndex;
  nvvk::Allocator* m_alloc{nullptr};
};
