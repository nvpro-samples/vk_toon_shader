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

#include "post_effect.hpp"

// Extract the contour from the normal and depth buffer
// And does a second post effect (FXAA) on the contour
class PostNrmDepth : public PostEffect
{
public:
  const std::string getShaderName() override { return R"(spv/contour_normaldepth.frag.spv)"; }

  void setup(const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t queueIndex, nvvk::ResourceAllocator* allocator) override
  {
    m_fxaa.setup(device, physicalDevice, queueIndex, allocator);
    PostEffect::setup(device, physicalDevice, queueIndex, allocator);
  }

  void initialize(const vk::Extent2D& size) override
  {
    m_fxaa.initialize(size);
    PostEffect::initialize(size);
  }

  void setInputs(const std::vector<nvvk::Texture>& inputs, const nvvk::Buffer& minMaxBuffer)
  {
    m_fxaa.setInputs({m_output});

    std::vector<vk::WriteDescriptorSet> writes;
    writes.emplace_back(m_descSetBind.makeWrite(m_descSet, 0, &inputs[0].descriptor));  // ray tracing
    vk::DescriptorBufferInfo bufInfo{minMaxBuffer.buffer, 0, VK_WHOLE_SIZE};
    writes.emplace_back(m_descSetBind.makeWrite(m_descSet, 1, &bufInfo));  // zNear - zFar from compute shader
    m_device.updateDescriptorSets(writes, nullptr);
  }

  void updateRenderTarget(const vk::Extent2D& size) override
  {
    m_fxaa.updateRenderTarget(size);
    PostEffect::updateRenderTarget(size);
  }

  void execute(const vk::CommandBuffer& cmdBuf) override
  {
    cmdBuf.pushConstants<PushConstant>(m_pipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, m_pushCnt);
    PostEffect::execute(cmdBuf);
    if(m_useFxaa)
      m_fxaa.execute(cmdBuf);
  }

  const nvvk::Texture getOutput() override
  {
    if(m_useFxaa)
      return m_fxaa.getOutput();

    return m_output;
  }

  void destroy() override
  {
    m_fxaa.destroy();
    PostEffect::destroy();
  }

  bool uiSetup() override
  {
    bool changed{false};
    changed |= ImGui::SliderFloat("Normal Threshold", &m_pushCnt.normalDiffCoeff, 0.0f, 1.f, "%.3f", ImGuiSliderFlags_Logarithmic);
    changed |= ImGui::SliderFloat("Depth Threshold", &m_pushCnt.depthDiffCoeff, 0.00f, 10.f, "%.3f", ImGuiSliderFlags_Logarithmic);
    changed |= ImGui::Checkbox("FXAA on Inside Details", &m_useFxaa);

    return changed;
  }

private:
  // One input image and push constant to control the effect
  void createDescriptorSet() override
  {
    vk::PushConstantRange push_constants = {vk::ShaderStageFlagBits::eFragment, 0, sizeof(PushConstant)};

    m_descSetBind.addBinding(vkDS(0, vkDT::eCombinedImageSampler, 1, vkSS::eFragment));  // Normal/depth from ray tracing
    m_descSetBind.addBinding(vkDS(1, vkDT::eStorageBuffer, 1, vkSS::eFragment));         // Min/Max

    m_descSetLayout  = m_descSetBind.createLayout(m_device);
    m_descPool       = m_descSetBind.createPool(m_device);
    m_descSet        = nvvk::allocateDescriptorSet(m_device, m_descPool, m_descSetLayout);
    m_pipelineLayout = m_device.createPipelineLayout({{}, 1, &m_descSetLayout, 1, &push_constants});
    m_debug.setObjectName(m_pipelineLayout, "nrmdepth");
  }

  struct PushConstant
  {
    float normalDiffCoeff{0.5f};
    float depthDiffCoeff{1.f};
  };
  PushConstant m_pushCnt;

  // Second post effect to anti-alias lines
  struct PostFxaa : public PostEffect
  {
    const std::string getShaderName() override { return R"(spv/fxaa.frag.spv)"; }
  };

  PostFxaa m_fxaa;
  bool     m_useFxaa{true};
};
