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

#include "imgui.h"
#include "nvh/fileoperations.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/images_vk.hpp"
#include "nvvk/pipeline_vk.hpp"

#include <algorithm>


extern std::vector<std::string> defaultSearchPaths;

// Base Class to create effect on incoming images and output one image
//
// Usage:
// - setup(...)
// - initialize( size of window/image )
// - setInput( image.descriptor )
// - run
// - getOutput
class PostEffect
{


public:
  PostEffect() = default;

  virtual void setup(const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t queueIndex, nvvk::Allocator* allocator)
  {
    m_device     = device;
    m_queueIndex = queueIndex;
    m_alloc      = allocator;
    m_debug.setup(device);
  }

  virtual void initialize(const VkExtent2D& size)
  {
    createRenderPass();
    createDescriptorSet();
    createPipeline();
    updateRenderTarget(size);
  }

  // Attaching the input image
  virtual void setInputs(const std::vector<nvvk::Texture>& inputs)
  {
    std::vector<vk::WriteDescriptorSet> writes;
    writes.emplace_back(m_descSetBind.makeWrite(m_descSet, 0, &inputs[0].descriptor));  // ray tracing
    m_device.updateDescriptorSets(writes, nullptr);
  }

  // Internal image format
  virtual vk::Format getOutputFormat() { return vk::Format::eR8G8B8A8Unorm; }
  // Display controls for the post-process
  virtual bool              uiSetup() { return false; }  // return true when something changed
  virtual const std::string getShaderName() = 0;


  void setActive(bool active_) { m_active = active_; }
  bool isActive() { return m_active; }

  // Updating the output framebuffer when the image size is changing
  virtual void updateRenderTarget(const VkExtent2D& size)
  {
    m_size = size;

    // Create new output image
    m_alloc->destroy(m_output);
    vk::SamplerCreateInfo samplerCreateInfo;  // default values
    vk::ImageCreateInfo   imageCreateInfo =
        nvvk::makeImage2DCreateInfo(m_size, getOutputFormat(),
                                    vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled);

    nvvk::Image             image  = m_alloc->createImage(imageCreateInfo);
    vk::ImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, imageCreateInfo);
    m_output                       = m_alloc->createTexture(image, ivInfo, samplerCreateInfo);

    {
      nvvk::CommandPool scb(m_device, m_queueIndex);
      auto              cmdBuf = scb.createCommandBuffer();
      nvvk::cmdBarrierImageLayout(cmdBuf, m_output.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal);
      m_alloc->finalizeAndReleaseStaging();
    }


    // Create the Framebuffer and attach the texture
    m_device.destroy(m_framebuffer);
    vk::FramebufferCreateInfo info;
    info.setRenderPass(m_renderPass);
    info.setAttachmentCount(1);
    info.setPAttachments(reinterpret_cast<vk::ImageView*>(&m_output.descriptor.imageView));
    info.setWidth(m_size.width);
    info.setHeight(m_size.height);
    info.setLayers(1);
    m_framebuffer = m_device.createFramebuffer(info);
  }

  virtual void destroy()
  {
    m_alloc->destroy(m_output);
    m_device.destroyFramebuffer(m_framebuffer);
    m_device.destroyDescriptorSetLayout(m_descSetLayout);
    m_device.destroyRenderPass(m_renderPass);
    m_device.destroyPipeline(m_pipeline);
    m_device.destroyPipelineLayout(m_pipelineLayout);
    m_device.destroyDescriptorPool(m_descPool);
  }

  virtual const nvvk::Texture getOutput() { return m_output; }


  // Executing the the post-process
  virtual void execute(const vk::CommandBuffer& cmdBuf)
  {
    vk::ClearValue clearValues;  // default is 0,0,0,0
    //clearValues[0].setColor(std::array<float,4>({0.f, 0.f, 0.f, 0.f}));

    vk::RenderPassBeginInfo renderPassBeginInfo = {m_renderPass, m_framebuffer, {{}, m_size}, 1, &clearValues};
    cmdBuf.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);
    if(m_active)
    {
      cmdBuf.setViewport(0, {vk::Viewport(0, 0, m_size.width, m_size.height, 0, 1)});
      cmdBuf.setScissor(0, {{{0, 0}, {m_size.width, m_size.height}}});
      cmdBuf.bindPipeline(vk::PipelineBindPoint::eGraphics, m_pipeline);
      cmdBuf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_pipelineLayout, 0, m_descSet, {});
      cmdBuf.draw(3, 1, 0, 0);
    }
    cmdBuf.endRenderPass();
  }

protected:
  // Render pass, one clear, no depth
  void createRenderPass()
  {
    if(m_renderPass)
      m_device.destroyRenderPass(m_renderPass);

    // Color attachment
    vk::AttachmentDescription attachments;
    attachments.setFormat(getOutputFormat());  // image format of the output image
    attachments.setLoadOp(vk::AttachmentLoadOp::eClear);
    attachments.setFinalLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

    vk::AttachmentReference colorReference{0, vk::ImageLayout::eColorAttachmentOptimal};
    vk::SubpassDescription  subpassDescription;
    subpassDescription.setColorAttachmentCount(1);
    subpassDescription.setPColorAttachments(&colorReference);

    vk::RenderPassCreateInfo renderPassInfo{{}, 1, &attachments, 1, &subpassDescription};
    m_renderPass = m_device.createRenderPass(renderPassInfo);

    std::string rp_name = getShaderName();
    rp_name = rp_name.substr(rp_name.find_first_of("/") + 1, rp_name.find_first_of(".") - rp_name.find_first_of("/") - 1);
    m_debug.setObjectName(m_renderPass, rp_name.c_str());
  }

  // One input image and push constant to control the effect
  virtual void createDescriptorSet()
  {
    m_descSetBind.addBinding(vkDS(0, vkDT::eCombinedImageSampler, 1, vkSS::eFragment));  // Normal/depth from ray tracing
    m_descSetLayout  = m_descSetBind.createLayout(m_device);
    m_descPool       = m_descSetBind.createPool(m_device);
    m_descSet        = nvvk::allocateDescriptorSet(m_device, m_descPool, m_descSetLayout);
    m_pipelineLayout = m_device.createPipelineLayout({{}, 1, &m_descSetLayout, 0, nullptr});
    m_debug.setObjectName(m_pipelineLayout, "post_effect");
  }


  // Creating the shading pipeline
  void createPipeline()
  {
    const std::string& fragProg = getShaderName();
    // Pipeline: completely generic, no vertices
    nvvk::GraphicsPipelineGeneratorCombined pipelineGenerator(m_device, m_pipelineLayout, m_renderPass);
    pipelineGenerator.addShader(nvh::loadFile("shaders/passthrough.vert.spv", true, defaultSearchPaths),
                                vk::ShaderStageFlagBits::eVertex);
    pipelineGenerator.addShader(nvh::loadFile(fragProg, true, defaultSearchPaths), vk::ShaderStageFlagBits::eFragment);
    pipelineGenerator.rasterizationState.setCullMode(vk::CullModeFlagBits::eNone);
    m_pipeline = pipelineGenerator.createPipeline();
  }

  //
  bool                        m_active{true};
  vk::RenderPass              m_renderPass;
  vk::Pipeline                m_pipeline;
  vk::PipelineLayout          m_pipelineLayout;
  nvvk::DescriptorSetBindings m_descSetBind;
  vk::DescriptorPool          m_descPool;
  vk::DescriptorSetLayout     m_descSetLayout;
  vk::DescriptorSet           m_descSet;
  vk::Extent2D                m_size{0, 0};
  vk::Framebuffer             m_framebuffer;
  nvvk::Texture               m_output;
  vk::Device                  m_device;
  uint32_t                    m_queueIndex;
  nvvk::Allocator*            m_alloc{nullptr};
  nvvk::DebugUtil             m_debug;
};
