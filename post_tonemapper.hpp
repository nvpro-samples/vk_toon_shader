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

// Take as an input an image (RGB32F) and apply a tonemapper
class Tonemapper : public PostEffect
{
public:
  const std::string getShaderName() override { return "shaders/tonemap.frag.spv"; }

  // Executing the the tonemapper
  void execute(const vk::CommandBuffer& cmdBuf) override
  {
    if(!m_active)
      return;

    cmdBuf.pushConstants<PushConstant>(m_pipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, m_pushCnt);
    PostEffect::execute(cmdBuf);
  }

  // Controlling the tonemapper
  bool uiSetup() override
  {
    static const std::vector<char const*> tmItem = {"Linear", "Uncharted 2", "Hejl Richard", "ACES"};
    bool                                  changed{false};
    changed |= ImGui::Combo("Tonemapper", &m_pushCnt.tonemapper, tmItem.data(), static_cast<int>(tmItem.size()));
    changed |= ImGui::InputFloat("Exposure", &m_pushCnt.exposure, 0.1f, 1.f);
    changed |= ImGui::InputFloat("Gamma", &m_pushCnt.gamma, .1f, 1.f);
    m_pushCnt.exposure = std::max(0.1f, std::min(m_pushCnt.exposure, 100.0f));
    m_pushCnt.gamma    = std::max(0.1f, std::min(m_pushCnt.gamma, 3.0f));
    return changed;
  }

private:
  // One input image and push constant to control the effect
  void createDescriptorSet() override
  {
    vk::PushConstantRange push_constants = {vk::ShaderStageFlagBits::eFragment, 0, sizeof(PushConstant)};
    m_descSetBind.clear();
    m_descSetBind.addBinding(vkDS(0, vkDT::eCombinedImageSampler, 1, vkSS::eFragment));  // Normal/depth from ray tracing
    m_descSetLayout  = m_descSetBind.createLayout(m_device);
    m_descPool       = m_descSetBind.createPool(m_device);
    m_descSet        = nvvk::allocateDescriptorSet(m_device, m_descPool, m_descSetLayout);
    m_pipelineLayout = m_device.createPipelineLayout({{}, 1, &m_descSetLayout, 1, &push_constants});
    m_debug.setObjectName(m_pipelineLayout, "tonemap");
  }

  struct PushConstant
  {
    int   tonemapper{1};
    float gamma{2.2f};
    float exposure{3.0f};
  };
  PushConstant m_pushCnt;
};
