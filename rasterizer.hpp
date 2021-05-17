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



#include <array>
#include <nvmath/nvmath.h>

#include "nvh/gltfscene.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "vk_util.hpp"


//--------------------------------------------------------------------------------------------------
// Simple example showing a cube, camera movement and post-process
//
class Rasterizer
{
public:
  Rasterizer() = default;

  void setup(const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t queueIndex, nvvk::ResourceAllocator* allocator);
  void setObjectPointers(nvh::GltfScene* gltfScene,
                         nvvk::Buffer*   vertexBuffer,
                         nvvk::Buffer*   normalBuffer,
                         nvvk::Buffer*   uvBuffer,
                         nvvk::Buffer*   indexBuffer);

  // Executing the rasterizer
  void run(const vk::CommandBuffer& cmdBuf, const vk::DescriptorSet& dsetScene, int frame = 0);

  // Return the rendered image
  const std::vector<nvvk::Texture>& outputImages() const;

  void destroy();
  void recordCommandBuffer(const vk::CommandPool& cmdPool, const vk::DescriptorSet& dsetScene);
  void createOutputImages(vk::Extent2D size);
  void createFrameBuffer();
  void createRenderPass();
  void createPipeline(const vk::DescriptorSetLayout& sceneDescSetLayout);

  void setClearColor(nvmath::vec3f& _color) { m_clearColor = _color; }
  void setToonSteps(int nbStep);
  void setToonLightDir(nvmath::vec3f lightDir);

private:
  void createDepthBuffer(vk::CommandBuffer commandBuffer, vk::Extent2D imageSize);
  void setViewport(const vk::CommandBuffer& cmdBuf);
  void render(const vk::CommandBuffer& cmdBuff, const vk::DescriptorSet& dsetScene);


  struct PushC
  {
    nvmath::vec3f lightDir{-1, -1, -1};
    int           nbSteps{5};
    int           instID{0};
    int           matID{0};
  } m_pushC;

  nvmath::vec3f m_clearColor{0, 0, 0};

  // Rasterizer
  vk::PipelineLayout         m_pipelineLayout;
  vk::Pipeline               m_drawPipeline;
  vk::CommandBuffer          m_recordedCmdBuffer;
  std::vector<nvvk::Texture> m_rasterizerOutput;  // many outputs (2)
  nvvk::Image                m_depthImage;
  vk::ImageView              m_depthImageView;
  vk::Extent2D               m_outputSize;
  vk::RenderPass             m_renderPass;
  vk::Framebuffer            m_framebuffer;
  vk::Format                 m_depthFormat{vk::Format::eD24UnormS8Uint};

  // Scene data
  nvh::GltfScene* m_gltfScene{nullptr};  // The scene
  nvvk::Buffer*   m_vertexBuffer{nullptr};
  nvvk::Buffer*   m_normalBuffer{nullptr};
  nvvk::Buffer*   m_uvBuffer{nullptr};
  nvvk::Buffer*   m_indexBuffer{nullptr};

  // Vulkan core
  vk::Device       m_device;
  nvvk::DebugUtil  m_debug;
  uint32_t         m_queueIndex;
  nvvk::ResourceAllocator* m_alloc{nullptr};
};
