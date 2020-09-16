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

#include "post_effect.hpp"

// Extract the contour from the normal and depth buffer
// And does a second post effect (FXAA) on the contour
class PostNrmDepth : public PostEffect
{
public:
  const std::string getShaderName() override { return R"(shaders/contour_normaldepth.frag.spv)"; }

  void setup(const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t queueIndex, nvvk::Allocator* allocator) override
  {
    m_fxaa.setup(device, physicalDevice, queueIndex, allocator);
    PostEffect::setup(device, physicalDevice, queueIndex, allocator);
  }

  void initialize(const VkExtent2D& size) override
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

  void updateRenderTarget(const VkExtent2D& size) override
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
    changed |= ImGui::SliderFloat("Normal Threshold", &m_pushCnt.normalDiffCoeff, 0.0f, 1.f, "%.3f", 2.0f);
    changed |= ImGui::SliderFloat("Depth Threshold", &m_pushCnt.depthDiffCoeff, 0.00f, 10.f, "%.3f", 5.0f);
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
    const std::string getShaderName() override { return R"(shaders/fxaa.frag.spv)"; }
  };

  PostFxaa m_fxaa;
  bool     m_useFxaa{true};
};
