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
// Raytracing implementation for the Vulkan Interop (G-Buffers)
//////////////////////////////////////////////////////////////////////////


#include "nvh/fileoperations.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/shaders_vk.hpp"
#include "vk_util.hpp"


extern std::vector<std::string> defaultSearchPaths;


struct RayPicker
{
private:
  std::vector<vk::RayTracingShaderGroupCreateInfoNV> m_groups;

public:
  struct PushConstant
  {
    float pickX{0};
    float pickY{0};
  } m_pushC;

  struct PickResult
  {
    nvmath::vec4f worldPos{0, 0, 0, 0};
    nvmath::vec4f barycentrics{0, 0, 0, 0};
    uint32_t      intanceID{0};
    uint32_t      intanceCustomID{0};
    uint32_t      primitiveID{0};
  };

  nvvk::Buffer m_pickResult;
  nvvk::Buffer m_sbtBuffer;

  nvvk::DescriptorSetBindings m_binding;

  vk::DescriptorPool                       m_descPool;
  vk::DescriptorSetLayout                  m_descSetLayout;
  vk::DescriptorSet                        m_descSet;
  vk::PipelineLayout                       m_pipelineLayout;
  vk::Pipeline                             m_pipeline;
  vk::PhysicalDeviceRayTracingPropertiesNV m_raytracingProperties;
  vk::AccelerationStructureNV              m_tlas;
  vk::PhysicalDevice                       m_physicalDevice;
  vk::Device                               m_device;
  uint32_t                                 m_queueIndex;
  nvvk::ResourceAllocator*                         m_alloc{nullptr};
  nvvk::DebugUtil                          m_debug;

  RayPicker() = default;


  void setup(const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t queueIndex, nvvk::ResourceAllocator* allocator)
  {
    m_physicalDevice = physicalDevice;
    m_device         = device;
    m_queueIndex     = queueIndex;
    m_debug.setup(device);
    m_alloc = allocator;
  }


  VkBuffer outputResult() const { return m_pickResult.buffer; }

  void destroy()
  {
    m_alloc->destroy(m_pickResult);
    m_alloc->destroy(m_sbtBuffer);
    m_device.destroyDescriptorSetLayout(m_descSetLayout);
    m_device.destroyPipelineLayout(m_pipelineLayout);
    m_device.destroyPipeline(m_pipeline);
    m_device.destroyDescriptorPool(m_descPool);
  }

  void initialize(const vk::AccelerationStructureNV& tlas, const vk::DescriptorBufferInfo& sceneUbo)
  {

    m_tlas = tlas;

    // Query the values of shaderHeaderSize and maxRecursionDepth in current implementation
    m_raytracingProperties =
        m_physicalDevice.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceRayTracingPropertiesNV>()
            .get<vk::PhysicalDeviceRayTracingPropertiesNV>();


    createOutputResult();
    createDescriptorSet(sceneUbo);
    createPipeline();
    createShadingBindingTable();
  }

  void createOutputResult()
  {
    m_alloc->destroy(m_pickResult);
    m_pickResult =
        m_alloc->createBuffer(sizeof(PickResult), vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eStorageBuffer,
                              vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    m_debug.setObjectName(m_pickResult.buffer, "PickResult");
  }

  void createDescriptorSet(const vk::DescriptorBufferInfo& sceneUbo)
  {
    m_binding.clear();
    m_binding.addBinding(vkDS(0, vkDT::eAccelerationStructureNV, 1, vkSS::eRaygenNV | vkSS::eClosestHitNV));
    m_binding.addBinding(vkDS(1, vkDT::eStorageBuffer, 1, vkSS::eRaygenNV));
    m_binding.addBinding(vkDS(2, vkDT::eUniformBuffer, 1, vkSS::eRaygenNV | vkSS::eClosestHitNV));

    m_descPool      = m_binding.createPool(m_device);
    m_descSetLayout = m_binding.createLayout(m_device);
    m_descSet       = m_device.allocateDescriptorSets({m_descPool, 1, &m_descSetLayout})[0];

    vk::WriteDescriptorSetAccelerationStructureNV descAsInfo{1, &m_tlas};

    vk::DescriptorBufferInfo            pickDesc{m_pickResult.buffer, 0, VK_WHOLE_SIZE};
    std::vector<vk::WriteDescriptorSet> writes;
    writes.emplace_back(m_binding.makeWrite(m_descSet, 0, &descAsInfo));
    writes.emplace_back(m_binding.makeWrite(m_descSet, 1, &pickDesc));
    writes.emplace_back(m_binding.makeWrite(m_descSet, 2, &sceneUbo));
    m_device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
  }


  void createPipeline()
  {
    vk::ShaderModule raygenSM = nvvk::createShaderModule(m_device, nvh::loadFile("spv/pick.rgen.spv", true, defaultSearchPaths));
    vk::ShaderModule missSM = nvvk::createShaderModule(m_device, nvh::loadFile("spv/pick.rmiss.spv", true, defaultSearchPaths));
    vk::ShaderModule chitSM = nvvk::createShaderModule(m_device, nvh::loadFile("spv/pick.rchit.spv", true, defaultSearchPaths));

    std::vector<vk::PipelineShaderStageCreateInfo> stages;

    // Raygen
    stages.push_back({{}, vk::ShaderStageFlagBits::eRaygenNV, raygenSM, "main"});
    vk::RayTracingShaderGroupCreateInfoNV rg{vk::RayTracingShaderGroupTypeNV::eGeneral, VK_SHADER_UNUSED_NV,
                                             VK_SHADER_UNUSED_NV, VK_SHADER_UNUSED_NV, VK_SHADER_UNUSED_NV};
    rg.setGeneralShader(static_cast<uint32_t>(stages.size() - 1));
    m_groups.push_back(rg);
    // Miss - TODO remove
    stages.push_back({{}, vk::ShaderStageFlagBits::eMissNV, missSM, "main"});
    vk::RayTracingShaderGroupCreateInfoNV mg{vk::RayTracingShaderGroupTypeNV::eGeneral, VK_SHADER_UNUSED_NV,
                                             VK_SHADER_UNUSED_NV, VK_SHADER_UNUSED_NV, VK_SHADER_UNUSED_NV};
    mg.setGeneralShader(static_cast<uint32_t>(stages.size() - 1));
    m_groups.push_back(mg);
    // Hit
    stages.push_back({{}, vk::ShaderStageFlagBits::eClosestHitNV, chitSM, "main"});
    vk::RayTracingShaderGroupCreateInfoNV hg{vk::RayTracingShaderGroupTypeNV::eTrianglesHitGroup, VK_SHADER_UNUSED_NV,
                                             VK_SHADER_UNUSED_NV, VK_SHADER_UNUSED_NV, VK_SHADER_UNUSED_NV};
    hg.setClosestHitShader(static_cast<uint32_t>(stages.size() - 1));
    m_groups.push_back(hg);

    vk::PushConstantRange        pushConstant{vk::ShaderStageFlagBits::eRaygenNV, 0, sizeof(PushConstant)};
    vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;
    pipelineLayoutCreateInfo.setSetLayoutCount(1);
    pipelineLayoutCreateInfo.setPSetLayouts(&m_descSetLayout);
    pipelineLayoutCreateInfo.setPushConstantRangeCount(1);
    pipelineLayoutCreateInfo.setPPushConstantRanges(&pushConstant);
    m_pipelineLayout = m_device.createPipelineLayout(pipelineLayoutCreateInfo);
    m_debug.setObjectName(m_pipelineLayout, "raypick");

    // Assemble the shader stages and recursion depth info into the raytracing pipeline
    vk::RayTracingPipelineCreateInfoNV rayPipelineInfo;
    rayPipelineInfo.setStageCount(static_cast<uint32_t>(stages.size()));
    rayPipelineInfo.setPStages(stages.data());
    rayPipelineInfo.setGroupCount(static_cast<uint32_t>(m_groups.size()));
    rayPipelineInfo.setPGroups(m_groups.data());
    rayPipelineInfo.setMaxRecursionDepth(2);
    rayPipelineInfo.setLayout(m_pipelineLayout);
    m_pipeline = static_cast<const vk::Pipeline&>(m_device.createRayTracingPipelineNV({}, rayPipelineInfo));

    m_device.destroyShaderModule(raygenSM);
    m_device.destroyShaderModule(missSM);
    m_device.destroyShaderModule(chitSM);
  }


  void createShadingBindingTable()
  {
    auto     groupCount      = static_cast<uint32_t>(m_groups.size());           // 3 shaders: raygen, miss, chit
    uint32_t groupHandleSize = m_raytracingProperties.shaderGroupHandleSize;     // Size of a program identifier
    uint32_t alignSize       = m_raytracingProperties.shaderGroupBaseAlignment;  // Size of a program identifier


    // Fetch all the shader handles used in the pipeline, so that they can be written in the SBT
    uint32_t             sbtSize = groupCount * alignSize;
    std::vector<uint8_t> shaderHandleStorage(sbtSize);
    auto res = m_device.getRayTracingShaderGroupHandlesNV(m_pipeline, 0, groupCount, sbtSize, shaderHandleStorage.data());
    assert(res == vk::Result::eSuccess);

    m_sbtBuffer = m_alloc->createBuffer(sbtSize, vk::BufferUsageFlagBits::eTransferSrc,
                                        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    m_debug.setObjectName(m_sbtBuffer.buffer, std::string("PickSBT").c_str());

    // Write the handles in the SBT
    void* mapped = m_alloc->map(m_sbtBuffer);
    auto* pData  = reinterpret_cast<uint8_t*>(mapped);
    for(uint32_t g = 0; g < groupCount; g++)
    {
      memcpy(pData, shaderHandleStorage.data() + g * groupHandleSize, groupHandleSize);  // raygen
      pData += alignSize;
    }
    m_alloc->unmap(m_sbtBuffer);
  }

  void run(const vk::CommandBuffer& cmdBuf, float x, float y)
  {
    m_pushC.pickX = x;
    m_pushC.pickY = y;

    uint32_t progSize = m_raytracingProperties.shaderGroupBaseAlignment;  // Size of a program identifier
    cmdBuf.bindPipeline(vk::PipelineBindPoint::eRayTracingNV, m_pipeline);
    cmdBuf.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingNV, m_pipelineLayout, 0, {m_descSet}, {});
    cmdBuf.pushConstants<PushConstant>(m_pipelineLayout, vk::ShaderStageFlagBits::eRaygenNV, 0, m_pushC);

    vk::DeviceSize rayGenOffset   = 0 * progSize;
    vk::DeviceSize missOffset     = 1 * progSize;
    vk::DeviceSize missStride     = progSize;
    vk::DeviceSize hitGroupOffset = 2 * progSize;  // Jump over the miss
    vk::DeviceSize hitGroupStride = progSize;

    cmdBuf.traceRaysNV(m_sbtBuffer.buffer, rayGenOffset,                    //
                       m_sbtBuffer.buffer, missOffset, missStride,          //
                       m_sbtBuffer.buffer, hitGroupOffset, hitGroupStride,  //
                       m_sbtBuffer.buffer, 0, 0,                            //
                       1, 1,                                                //
                       1 /*, NVVKPP_DISPATCHER*/);

    vk::BufferMemoryBarrier bmb{vk::AccessFlagBits::eMemoryWrite,
                                vk::AccessFlagBits::eMemoryRead,
                                VK_QUEUE_FAMILY_IGNORED,
                                VK_QUEUE_FAMILY_IGNORED,
                                m_pickResult.buffer,
                                0,
                                VK_WHOLE_SIZE};
    cmdBuf.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTopOfPipe,
                           vk::DependencyFlagBits::eDeviceGroup, 0, nullptr, 1, &bmb, 0, nullptr);
  }

  PickResult getResult()
  {
    PickResult pr;
    void*      mapped = m_alloc->map(m_pickResult);
    memcpy(&pr, mapped, sizeof(PickResult));
    m_alloc->unmap(m_pickResult);
    return pr;
  }
};
