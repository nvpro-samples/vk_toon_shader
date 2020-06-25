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

// This is the implementation of the anisotropic Kuwahara filter (https://hpi.de/en/doellner/rendering/akf.html)
// Paper : http://www.kyprianidis.com/p/pg2009/jkyprian-pg2009.pdf
// Preview App' : https://code.google.com/archive/p/gpuakf/downloads
// Code : https://code.google.com/archive/p/gpuakf/source/default/source

// Apply anisotropic Kuwahara filter (https://hpi.de/en/doellner/rendering/akf.html)
class PostKuwaharaAniso : public PostEffect
{
public:
  const std::string getShaderName() override { return R"(shaders/kuwa_aniso.frag.spv)"; }

  void setup(const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t queueIndex, nvvk::Allocator* allocator) override
  {
    m_sst.setup(device, physicalDevice, queueIndex, allocator);
    m_gauss.setup(device, physicalDevice, queueIndex, allocator);
    m_tfm.setup(device, physicalDevice, queueIndex, allocator);
    PostEffect::setup(device, physicalDevice, queueIndex, allocator);
  }

  void initialize(const VkExtent2D& size) override
  {
    m_sst.initialize(size);
    m_gauss.initialize(size);
    m_tfm.initialize(size);
    PostEffect::initialize(size);
    updateKernel();
  }

  void setInputs(const std::vector<nvvk::Texture>& inputs) override
  {
    m_sst.setInputs(inputs);
    m_gauss.setInputs({m_sst.getOutput()});
    m_tfm.setInputs({m_gauss.getOutput()});

    const nvvk::Texture& outTfm = m_tfm.getOutput();

    std::vector<vk::WriteDescriptorSet> writes;
    writes.emplace_back(m_descSetBind.makeWrite(m_descSet, 0, &inputs[0].descriptor));  // ray tracing
    writes.emplace_back(m_descSetBind.makeWrite(m_descSet, 1, &m_kernel.descriptor));   // kernel
    writes.emplace_back(m_descSetBind.makeWrite(m_descSet, 2, &outTfm.descriptor));     // kuwahara info
    m_device.updateDescriptorSets(writes, nullptr);
  }

  void updateRenderTarget(const VkExtent2D& size) override
  {
    m_sst.updateRenderTarget(size);
    m_gauss.updateRenderTarget(size);
    m_tfm.updateRenderTarget(size);
    PostEffect::updateRenderTarget(size);
  }

  // Executing the effect
  void execute(const vk::CommandBuffer& cmdBuf) override
  {
    if(!m_active)
      return;

    m_sst.execute(cmdBuf);
    m_gauss.execute(cmdBuf);
    m_tfm.execute(cmdBuf);
    cmdBuf.pushConstants<PushConstant>(m_pipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, m_pushCnt);
    PostEffect::execute(cmdBuf);
  }

  void destroy() override
  {
    m_sst.destroy();
    m_gauss.destroy();
    m_tfm.destroy();
    m_alloc->destroy(m_kernel);
    PostEffect::destroy();
  }

  // UI Control
  bool uiSetup() override
  {
    bool changed{false};
    //    changed |= ImGui::InputFloat("alpha", &m_pushCnt.alpha);
    changed |= ImGui::SliderFloat("radius", &m_pushCnt.radius, 1, 20);
    changed |= ImGui::SliderInt("N", &m_Nsectors, 1, 16);
    if(changed)
      updateKernel();
    return changed;
  }

private:
  // One input image and push constant to control the effect
  void createDescriptorSet() override
  {
    vk::PushConstantRange push_constants = {vk::ShaderStageFlagBits::eFragment, 0, sizeof(PushConstant)};

    m_descSetBind.addBinding(vkDS(0, vkDT::eCombinedImageSampler, 1, vkSS::eFragment));  // ray tracing image
    m_descSetBind.addBinding(vkDS(1, vkDT::eCombinedImageSampler, 1, vkSS::eFragment));  // kernel
    m_descSetBind.addBinding(vkDS(2, vkDT::eCombinedImageSampler, 1, vkSS::eFragment));  // from tfm
    m_descSetLayout  = m_descSetBind.createLayout(m_device);
    m_descPool       = m_descSetBind.createPool(m_device);
    m_descSet        = nvvk::allocateDescriptorSet(m_device, m_descPool, m_descSetLayout);
    m_pipelineLayout = m_device.createPipelineLayout({{}, 1, &m_descSetLayout, 1, &push_constants});
    m_debug.setObjectName(m_pipelineLayout, "kuwahara_aniso");
  }


  void updateKernel()
  {
    int   N         = m_Nsectors;
    float smoothing = m_smoothing / 100.0f;  // in %

    const int   krnl_size = 32;
    const float sigma     = 0.25f * (krnl_size - 1);

    float* krnl[4];
    for(int k = 0; k < 4; ++k)
    {
      krnl[k] = new float[krnl_size * krnl_size];
      make_sector(krnl[k], k, N, krnl_size, sigma, smoothing * sigma);
    }

    m_device.waitIdle();
    m_alloc->destroy(m_kernel);
    {
      nvvk::ScopeCommandBuffer cmdBuf(m_device, m_queueIndex);
      vk::SamplerCreateInfo    samplerCreateInfo;  // default values
      vk::Extent2D             size(krnl_size, krnl_size);
      vk::ImageCreateInfo      imageCreateInfo = nvvk::makeImage2DCreateInfo(size, vk::Format::eR32Sfloat);

      nvvk::Image image = m_alloc->createImage(cmdBuf, krnl_size * krnl_size * sizeof(float), krnl[0], imageCreateInfo);
      vk::ImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, imageCreateInfo);
      m_kernel                       = m_alloc->createTexture(image, ivInfo, samplerCreateInfo);


      nvvk::cmdBarrierImageLayout(cmdBuf, m_kernel.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal);
    }
    m_alloc->finalizeAndReleaseStaging();
    m_debug.setObjectName(m_kernel.image, "Kernel");

    for(auto& k : krnl)
    {
      delete k;
    }
  }


  static void gauss_filter(float* data, int width, int height, float sigma)
  {
    float twoSigma2 = 2.0f * sigma * sigma;
    int   halfWidth = (int)ceil(2.0 * sigma);

    //float*             src_data = new float[width * height];
    std::vector<float> srcData(width * height);

    memcpy(srcData.data(), data, width * height * sizeof(float));

    for(int y = 0; y < height; ++y)
    {
      for(int x = 0; x < width; ++x)
      {
        float sum = 0;
        float w   = 0;

        for(int i = -halfWidth; i <= halfWidth; ++i)
        {
          for(int j = -halfWidth; j <= halfWidth; ++j)
          {
            int xi = x + i;
            int yj = y + j;
            if((xi >= 0) && (xi < width) && (yj >= 0) && (yj < height))
            {
              float r = sqrt((float)(i * i + j * j));
              float k = exp(-r * r / twoSigma2);
              w += k;
              sum += k * srcData[xi + yj * width];
            }
          }
        }

        data[x + y * width] = sum / w;
      }
    }

    //    delete[] src_data;
  }


  void make_sector(float* krnl, int k, int N, int size, float sigma_r, float sigma_s)
  {
    float* p = krnl;
    for(int j = 0; j < size; ++j)
    {
      for(int i = 0; i < size; ++i)
      {
        float x = i - 0.5f * size + 0.5f;
        float y = j - 0.5f * size + 0.5f;
        float r = sqrtf((x * x + y * y));

        float a = 0.5f * atan2(y, x) / float(nv_pi) + k * 1.0f / N;
        if(a > 0.5f)
          a -= 1.0f;
        if(a < -0.5f)
          a += 1.0f;

        if((fabs(a) <= 0.5f / N) && (r < 0.5f * size))
        {
          *p = 1;
        }
        else
        {
          *p = 0;
        }
        ++p;
      }
    }

    gauss_filter(krnl, size, size, sigma_s);

    p        = krnl;
    float mx = 0.0f;
    for(int j = 0; j < size; ++j)
    {
      for(int i = 0; i < size; ++i)
      {
        float x = i - 0.5f * size + 0.5f;
        float y = j - 0.5f * size + 0.5f;
        float r = sqrtf(x * x + y * y);
        *p *= exp(-0.5f * r * r / sigma_r / sigma_r);
        if(*p > mx)
          mx = *p;
        ++p;
      }
    }

    p = krnl;
    for(int j = 0; j < size; ++j)
    {
      for(int i = 0; i < size; ++i)
      {
        *p /= mx;
        ++p;
      }
    }
  }


  struct PushConstant
  {
    float radius{3.f};
    float q{8.f};
    float alpha{1.f};
  };
  PushConstant m_pushCnt;

  nvvk::Texture m_kernel;
  int           m_Nsectors  = 8;
  float         m_smoothing = 33.33f;
  float         m_sigma_t{2.0f};


  struct PostKSst : public PostEffect
  {
    const std::string getShaderName() override { return R"(shaders/kuwa_sst.frag.spv)"; }
  };

  struct PostKTfm : public PostEffect
  {
    const std::string getShaderName() override { return R"(shaders/kuwa_tfm.frag.spv)"; }
  };

  struct PostGauss : public PostEffect
  {
    const std::string getShaderName() override { return R"(shaders/kuwa_gauss.frag.spv)"; }
  };

  PostKSst  m_sst;
  PostKTfm  m_tfm;
  PostGauss m_gauss;
};
