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

// Take as an input an image (RGB32F) and apply a tonemapper
class Tonemapper : public PostEffect
{
public:
  const std::string getShaderName() override { return "spv/tonemap.frag.spv"; }

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
