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

// Merge the final frame and add the contour of the normal/depth and object
class PostCompositing : public PostEffect
{
public:
  const std::string getShaderName() override { return R"(shaders/compositing.frag.spv)"; }

  // Attaching the 3 input images
  void setInputs(const std::vector<nvvk::Texture>& inputs) override
  {
    std::vector<vk::WriteDescriptorSet> writes;
    writes.emplace_back(m_descSetBind.makeWrite(m_descSet, 0, &inputs[0].descriptor));  // ray tracing
    writes.emplace_back(m_descSetBind.makeWrite(m_descSet, 1, &inputs[1].descriptor));  // normal depth
    writes.emplace_back(m_descSetBind.makeWrite(m_descSet, 2, &inputs[2].descriptor));  // object
    m_device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
  }

  void execute(const vk::CommandBuffer& cmdBuf) override
  {
    if(!m_active)
      return;

    cmdBuf.pushConstants<PushConstant>(m_pipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, m_pushCnt);
    PostEffect::execute(cmdBuf);
  }

  bool uiSetup() override
  {
    bool changed{false};
    //    changed |= ImGui::Checkbox("Color Background", (bool*)&m_pushCnt.setBackground);
    //    changed |= ImGui::ColorEdit3("Background Color", &m_pushCnt.backgroundColor.x);
    changed |= ImGui::ColorEdit3("Contour Color", &m_pushCnt.lineColor.x);
    return changed;
  }

private:
  // One input image and push constant to control the effect
  void createDescriptorSet() override
  {
    vk::PushConstantRange push_constants = {vk::ShaderStageFlagBits::eFragment, 0, sizeof(PushConstant)};

    m_descSetBind.addBinding(vkDS(0, vkDT::eCombinedImageSampler, 1, vkSS::eFragment));  // ray tracing
    m_descSetBind.addBinding(vkDS(1, vkDT::eCombinedImageSampler, 1, vkSS::eFragment));  // Normal/depth
    m_descSetBind.addBinding(vkDS(2, vkDT::eCombinedImageSampler, 1, vkSS::eFragment));  // Object contour
    m_descSetLayout  = m_descSetBind.createLayout(m_device);
    m_descPool       = m_descSetBind.createPool(m_device);
    m_descSet        = nvvk::allocateDescriptorSet(m_device, m_descPool, m_descSetLayout);
    m_pipelineLayout = m_device.createPipelineLayout({{}, 1, &m_descSetLayout, 1, &push_constants});
    m_debug.setObjectName(m_pipelineLayout, "compositing");
  }

  struct PushConstant
  {
    nvmath::vec3f backgroundColor{1.f};
    int           setBackground{0};
    nvmath::vec3f lineColor{0.3f};
  };
  PushConstant m_pushCnt;
};
