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


//--------------------------------------------------------------------------------------------------
// This example is loading a glTF scene and renders it with a very simple material
//

#include <iostream>
#include <vulkan/vulkan.hpp>

#include "nvh/fileoperations.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/renderpasses_vk.hpp"
#include "nvvk/shaders_vk.hpp"
#include "rasterizer.hpp"


extern std::vector<std::string> defaultSearchPaths;

void Rasterizer::setup(const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t queueIndex, nvvk::Allocator* allocator)
{
  m_device     = device;
  m_queueIndex = queueIndex;
  m_debug.setup(device);
  m_alloc = allocator;
}

//--------------------------------------------------------------------------------------------------
// Overridden function called on shutdown
//
void Rasterizer::destroy()
{
  m_device.waitIdle();

  m_alloc->destroy(m_depthImage);
  for(auto& t : m_rasterizerOutput)
    m_alloc->destroy(t);

  m_device.destroy(m_renderPass);
  m_device.destroy(m_framebuffer);
  m_device.destroy(m_drawPipeline);
  m_device.destroy(m_pipelineLayout);
  m_device.destroy(m_depthImageView);

  m_framebuffer    = vk::Framebuffer();
  m_depthImageView = vk::ImageView();
}


//--------------------------------------------------------------------------------
// Called at each frame, as fast as possible
//
void Rasterizer::run(const vk::CommandBuffer& cmdBuf, const vk::DescriptorSet& dsetScene, int frame /*= 0*/)
{
  auto dbgLabel = m_debug.scopeLabel(cmdBuf, "Start rendering");

  vk::ClearValue clearValues[3];
  clearValues[0].setColor(makeClearColor(m_clearColor));                    // Color buffer
  clearValues[1].setColor(std::array<float, 4>({0.0f, 0.0f, -1.0f, 0.f}));  // Data buffer
  clearValues[2].setDepthStencil({1.0f, 0});


  // Pre-recorded scene
  {
    auto dbgLabel = m_debug.scopeLabel(cmdBuf, "Recorded Scene");

    vk::RenderPassBeginInfo renderPassBeginInfo{m_renderPass, m_framebuffer, {{}, m_outputSize}, 3, clearValues};
    // Recorded
    //cmdBuf.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eSecondaryCommandBuffers);
    //cmdBuf.executeCommands(m_recordedCmdBuffer);
    // Immediate
    cmdBuf.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);
    setViewport(cmdBuf);
    render(cmdBuf, dsetScene);
    cmdBuf.endRenderPass();
  }
}

//--------------------------------------------------------------------------------------------------
// When the pipeline is set for using dynamic, this becomes useful
//
void Rasterizer::setViewport(const vk::CommandBuffer& cmdBuf)
{
  cmdBuf.setViewport(0, {vk::Viewport(0.0f, 0.0f, static_cast<float>(m_outputSize.width),
                                      static_cast<float>(m_outputSize.height), 0.0f, 1.0f)});
  cmdBuf.setScissor(0, {{{0, 0}, {m_outputSize.width, m_outputSize.height}}});
}


//--------------------------------------------------------------------------------------------------
// Building the command buffer, is in fact, recording all the calls  needed to draw the frame in a
// command buffer.This need to be  call only if the number of objects in the scene is changing or
// if the viewport is changing
//
void Rasterizer::recordCommandBuffer(const vk::CommandPool& cmdPool, const vk::DescriptorSet& dsetScene)
{
  m_device.freeCommandBuffers(cmdPool, {m_recordedCmdBuffer});
  m_recordedCmdBuffer = m_device.allocateCommandBuffers({cmdPool, vk::CommandBufferLevel::eSecondary, 1})[0];

  vk::CommandBufferInheritanceInfo inheritance_info{m_renderPass};
  vk::CommandBufferBeginInfo       begin_info{vkCB::eSimultaneousUse | vkCB::eRenderPassContinue, &inheritance_info};

  m_recordedCmdBuffer.begin(begin_info);
  {
    setViewport(m_recordedCmdBuffer);
    render(m_recordedCmdBuffer, dsetScene);
  }
  m_recordedCmdBuffer.end();
}


//--------------------------------------------------------------------------------------------------
// The pipeline is how things are rendered, which shaders, type of primitives, depth test and more
//
void Rasterizer::createPipeline(const vk::DescriptorSetLayout& sceneDescSetLayout)
{
  vk::PushConstantRange pushConstantRanges = {vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0,
                                              sizeof(PushC)};

  // Creating the pipeline layout
  vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;
  pipelineLayoutCreateInfo.setSetLayoutCount(1);
  pipelineLayoutCreateInfo.setPSetLayouts(&sceneDescSetLayout);
  pipelineLayoutCreateInfo.setPushConstantRangeCount(1);
  pipelineLayoutCreateInfo.setPPushConstantRanges(&pushConstantRanges);
  m_pipelineLayout = m_device.createPipelineLayout(pipelineLayoutCreateInfo);

  // Pipeline
  std::vector<std::string>                paths = defaultSearchPaths;
  nvvk::GraphicsPipelineGeneratorCombined gpb(m_device, m_pipelineLayout, m_renderPass);
  gpb.depthStencilState.depthTestEnable = true;

  gpb.addBlendAttachmentState(nvvk::GraphicsPipelineState::makePipelineColorBlendAttachmentState());
  gpb.addShader(nvh::loadFile("shaders/rasterizer.vert.spv", true, paths), vk::ShaderStageFlagBits::eVertex);
  gpb.addShader(nvh::loadFile("shaders/rasterizer.frag.spv", true, paths), vk::ShaderStageFlagBits::eFragment);
  gpb.addBindingDescriptions({{0, sizeof(nvmath::vec3)}, {1, sizeof(nvmath::vec3)}, {2, sizeof(nvmath::vec2)}});
  gpb.addAttributeDescriptions({
      {0, 0, vk::Format::eR32G32B32Sfloat, 0},  // Position
      {1, 1, vk::Format::eR32G32B32Sfloat, 0},  // Normal
      {2, 2, vk::Format::eR32G32Sfloat, 0},     // Texcoord0
  });
  gpb.rasterizationState.setCullMode(vk::CullModeFlagBits::eNone);
  m_drawPipeline = gpb.createPipeline();

  m_debug.setObjectName(m_drawPipeline, "ShadingPipeline");
  m_debug.setObjectName(gpb.getShaderModule(0), "VertexShader");
  m_debug.setObjectName(gpb.getShaderModule(1), "FragmentShader");
}


//--------------------------------------------------------------------------------------------------
// Rendering all glTF nodes
//
void Rasterizer::render(const vk::CommandBuffer& cmdBuff, const vk::DescriptorSet& dsetScene)
{
  if(!m_drawPipeline)
  {
    return;
  }

  m_debug.setObjectName(cmdBuff, "Recored");
  auto dgbLabel = m_debug.scopeLabel(cmdBuff, "Recording Scene");

  // Pipeline to use for rendering the current scene
  cmdBuff.bindPipeline(vk::PipelineBindPoint::eGraphics, m_drawPipeline);

  // Offsets for the descriptor set and vertex buffer
  std::vector<vk::DeviceSize> offsets = {0, 0, 0};

  // Keeping track of the last material to avoid binding them again
  uint32_t lastMaterial = -1;

  std::vector<vk::Buffer> vertexBuffers = {m_vertexBuffer->buffer, m_normalBuffer->buffer, m_uvBuffer->buffer};
  cmdBuff.bindVertexBuffers(0, static_cast<uint32_t>(vertexBuffers.size()), vertexBuffers.data(), offsets.data());
  cmdBuff.bindIndexBuffer(m_indexBuffer->buffer, 0, vk::IndexType::eUint32);

  std::vector<vk::DescriptorSet> descriptorSets = {dsetScene};

  // The pipeline uses four descriptor set, one for the scene information, one for the matrix of the instance, one for the textures and for the environment
  cmdBuff.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_pipelineLayout, 0, descriptorSets, {});

  uint32_t idxNode = 0;
  for(auto& node : m_gltfScene->m_nodes)
  {
    auto  dgbLabel  = m_debug.scopeLabel(cmdBuff, std::string("Draw Mesh: " + std::to_string(node.primMesh)));
    auto& primitive = m_gltfScene->m_primMeshes[node.primMesh];

    m_pushC.instID = idxNode++;
    m_pushC.matID  = primitive.materialIndex;
    cmdBuff.pushConstants<PushC>(m_pipelineLayout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, m_pushC);
    cmdBuff.drawIndexed(primitive.indexCount, 1, primitive.firstIndex, primitive.vertexOffset, 0);
  }
}


void Rasterizer::setToonSteps(int nbStep)
{
  m_pushC.nbSteps = nbStep;
}

void Rasterizer::setToonLightDir(nvmath::vec3f lightDir)
{
  m_pushC.lightDir = lightDir;
}

// Return all outputs
const std::vector<nvvk::Texture>& Rasterizer::outputImages() const
{
  return m_rasterizerOutput;
}

//--------------------------------------------------------------------------------------------------
// The display will render the recorded command buffer, then in a sub-pass, render the UI
//
void Rasterizer::createRenderPass()
{
  m_renderPass = nvvk::createRenderPass(m_device, {vk::Format::eR32G32B32A32Sfloat, vk::Format::eR32G32B32A32Sfloat},  // color attachment
                                        m_depthFormat,                // depth attachment
                                        1,                            // Nb sub-passes
                                        true,                         // clearColor
                                        true,                         // clearDepth
                                        vk::ImageLayout::eUndefined,  // initialLayout
                                        vk::ImageLayout::eGeneral);   // finalLayout

  m_debug.setObjectName(m_renderPass, "General Render Pass");
}


//--------------------------------------------------------------------------------------------------
// Making the two output images: color, data(normal, depth, ID)
//
void Rasterizer::createOutputImages(vk::Extent2D size)
{
  for(auto& t : m_rasterizerOutput)
    m_alloc->destroy(t);
  m_rasterizerOutput.clear();

  m_outputSize = size;
  auto usage   = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage
               | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eColorAttachment;
  vk::DeviceSize imgSize = size.width * size.height * 4 * sizeof(float);
  vk::Format     format  = vk::Format::eR32G32B32A32Sfloat;

  // Create two output image, the color and the data
  for(int i = 0; i < 2; i++)
  {
    nvvk::ScopeCommandBuffer cmdBuf(m_device, m_queueIndex);
    vk::SamplerCreateInfo    samplerCreateInfo;  // default values
    vk::ImageCreateInfo      imageCreateInfo = nvvk::makeImage2DCreateInfo(size, format, usage);

    nvvk::Image image = m_alloc->createImage(cmdBuf, imgSize, nullptr, imageCreateInfo, vk::ImageLayout::eGeneral);
    vk::ImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, imageCreateInfo);
    nvvk::Texture           txt    = m_alloc->createTexture(image, ivInfo, samplerCreateInfo);
    txt.descriptor.imageLayout     = VK_IMAGE_LAYOUT_GENERAL;

    m_rasterizerOutput.push_back(txt);
  }
  {
    nvvk::ScopeCommandBuffer cmdBuf(m_device, m_queueIndex);
    createDepthBuffer(cmdBuf, size);
  }
  createFrameBuffer();
}

//--------------------------------------------------------------------------------------------------
// Create the framebuffers in which the images will be rendered
// - Swapchain need to be created before calling this
//
void Rasterizer::createFrameBuffer()
{
  // Recreate the frame buffers
  m_device.destroy(m_framebuffer);

  // Array of attachment (color, depth)
  std::array<vk::ImageView, 3> attachments;

  // Create frame buffers for every swap chain image
  vk::FramebufferCreateInfo framebufferCreateInfo;
  framebufferCreateInfo.renderPass      = m_renderPass;
  framebufferCreateInfo.attachmentCount = 3;
  framebufferCreateInfo.width           = m_outputSize.width;
  framebufferCreateInfo.height          = m_outputSize.height;
  framebufferCreateInfo.layers          = 1;
  framebufferCreateInfo.pAttachments    = attachments.data();

  // Create frame buffers for every swap chain image
  attachments[0] = m_rasterizerOutput[0].descriptor.imageView;  // Color
  attachments[1] = m_rasterizerOutput[1].descriptor.imageView;  // Data
  attachments[2] = m_depthImageView;                            // Depth
  m_framebuffer  = m_device.createFramebuffer(framebufferCreateInfo);

  std::string name = std::string("Rasterizer_Framebuffer");
#if DEBUG
  m_device.setDebugUtilsObjectNameEXT(
      {vk::ObjectType::eFramebuffer, reinterpret_cast<const uint64_t&>(m_framebuffer), name.c_str()});
#endif
}


//--------------------------------------------------------------------------------------------------
// Creating an image to be used as depth buffer
//
void Rasterizer::createDepthBuffer(vk::CommandBuffer commandBuffer, vk::Extent2D imageSize)
{
  m_alloc->destroy(m_depthImage);
  m_device.destroy(m_depthImageView);

  vk::ImageCreateInfo imageInfo =
      nvvk::makeImage2DCreateInfo(imageSize, m_depthFormat, vk::ImageUsageFlagBits::eDepthStencilAttachment);
  m_depthImage = m_alloc->createImage(imageInfo);
  m_debug.setObjectName(m_depthImage.image, "m_depthImage");

  vk::ImageViewCreateInfo viewInfo =
      nvvk::makeImage2DViewCreateInfo(m_depthImage.image, m_depthFormat, vk::ImageAspectFlagBits::eDepth);
  m_depthImageView = m_device.createImageView(viewInfo);
  m_debug.setObjectName(m_depthImageView, "m_depthImageView");

  // Set layout to VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
  nvvk::cmdBarrierImageLayout(commandBuffer,                                    // Command buffer
                              m_depthImage.image,                               // Image
                              vk::ImageLayout::eUndefined,                      // Old layout
                              vk::ImageLayout::eDepthStencilAttachmentOptimal,  // New layout
                              vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil);
}


void Rasterizer::setObjectPointers(nvh::GltfScene* gltfScene,
                                   nvvk::Buffer*   vertexBuffer,
                                   nvvk::Buffer*   normalBuffer,
                                   nvvk::Buffer*   uvBuffer,
                                   nvvk::Buffer*   indexBuffer)
{
  m_gltfScene    = gltfScene;
  m_vertexBuffer = vertexBuffer;
  m_normalBuffer = normalBuffer;
  m_uvBuffer     = uvBuffer;
  m_indexBuffer  = indexBuffer;
}
