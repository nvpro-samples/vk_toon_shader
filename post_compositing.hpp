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
#include "nvvk/resourceallocator_vk.hpp"

// Merge the final frame and add the contour of the normal/depth and object
class PostCompositing : public PostEffect
{
public:
  const std::string getShaderName() override { return R"(spv/compositing.frag.spv)"; }

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
    glm::vec3 backgroundColor{1.f};
    int       setBackground{0};
    glm::vec3 lineColor{0.3f};
  };
  PushConstant m_pushCnt;
};
