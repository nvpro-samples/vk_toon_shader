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

  void setup(const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t queueIndex, nvvk::Allocator* allocator);
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
  nvvk::Allocator* m_alloc{nullptr};
};
