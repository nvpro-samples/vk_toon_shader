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


#include "raytracer.hpp"
#include "imgui.h"
#include "nvh/fileoperations.hpp"
#include "nvvk/shaders_vk.hpp"

extern std::vector<std::string> defaultSearchPaths;

Raytracer::Raytracer() = default;

//--------------------------------------------------------------------------------------------------
// Initializing the allocator and querying the raytracing properties
//
void Raytracer::setup(const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t queueIndex, nvvk::Allocator* allocator)
{
  m_device     = device;
  m_queueIndex = queueIndex;
  m_debug.setup(device);
  m_alloc = allocator;

  // Requesting raytracing properties
  auto properties = physicalDevice.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceRayTracingPropertiesNV>();
  m_rtProperties = properties.get<vk::PhysicalDeviceRayTracingPropertiesNV>();

  if(m_rtProperties.shaderGroupHandleSize != 0)
    m_bValid = true;
  else
  {
    m_bValid = false;
    return;
  }
  m_rtBuilder.setup(device, allocator, queueIndex);
}

const std::vector<nvvk::Texture>& Raytracer::outputImages() const
{
  return m_raytracingOutput;
}

int Raytracer::maxFrames() const
{
  return m_maxFrames;
}

void Raytracer::destroy()
{
  for(auto& t : m_raytracingOutput)
    m_alloc->destroy(t);
  m_rtBuilder.destroy();
  m_device.destroy(m_descPool);
  m_device.destroy(m_descSetLayout);
  m_device.destroy(m_pipeline);
  m_device.destroy(m_pipelineLayout);
  m_alloc->destroy(m_sbtBuffer);
  m_alloc->destroy(m_rtPrimLookup);
  m_binding.clear();
}

//--------------------------------------------------------------------------------------------------
// Making all output images: color, normal, ...
//
void Raytracer::createOutputImages(vk::Extent2D size)
{
  for(auto& t : m_raytracingOutput)
    m_alloc->destroy(t);
  m_raytracingOutput.clear();

  m_outputSize = size;
  auto usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc;
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

    m_raytracingOutput.push_back(txt);
  }
  m_alloc->finalizeAndReleaseStaging();
}

void Raytracer::createDescriptorSet()
{
  using vkDS = vk::DescriptorSetLayoutBinding;
  using vkDT = vk::DescriptorType;
  using vkSS = vk::ShaderStageFlagBits;

  uint32_t nbOutput = static_cast<uint32_t>(m_raytracingOutput.size());

  m_binding.addBinding(vkDS(0, vkDT::eAccelerationStructureNV, 1, vkSS::eRaygenNV | vkSS::eClosestHitNV));
  m_binding.addBinding(vkDS(1, vkDT::eStorageImage, nbOutput, vkSS::eRaygenNV));                  // Output image
  m_binding.addBinding(vkDS(2, vkDT::eStorageBuffer, 1, vkSS::eClosestHitNV | vkSS::eAnyHitNV));  // Primitive info

  m_descPool      = m_binding.createPool(m_device);
  m_descSetLayout = m_binding.createLayout(m_device);
  m_descSet       = m_device.allocateDescriptorSets({m_descPool, 1, &m_descSetLayout})[0];

  std::vector<vk::WriteDescriptorSet> writes;

  vk::AccelerationStructureNV                   tlas = m_rtBuilder.getAccelerationStructure();
  vk::WriteDescriptorSetAccelerationStructureNV descAsInfo{1, &tlas};
  vk::DescriptorBufferInfo                      primitiveInfoDesc{m_rtPrimLookup.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_binding.makeWrite(m_descSet, 0, &descAsInfo));

  std::vector<vk::DescriptorImageInfo> descImgInfo;
  for(auto& i : m_raytracingOutput)
  {
    descImgInfo.push_back(i.descriptor);
  }
  writes.emplace_back(m_binding.makeWriteArray(m_descSet, 1, descImgInfo.data()));

  writes.emplace_back(m_binding.makeWrite(m_descSet, 2, &primitiveInfoDesc));
  m_device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

  updateDescriptorSet();
}

void Raytracer::updateDescriptorSet()
{
  // (1) Output buffer
  {
    std::vector<vk::DescriptorImageInfo> descImgInfo;
    for(auto& i : m_raytracingOutput)
    {
      descImgInfo.push_back(i.descriptor);
    }
    vk::WriteDescriptorSet wds = m_binding.makeWriteArray(m_descSet, 1, descImgInfo.data());

    //vk::DescriptorImageInfo imageInfo{{}, m_raytracingOutput.descriptor.imageView, vk::ImageLayout::eGeneral};
    //vk::WriteDescriptorSet  wds{m_rtDescSet, 1, 0, 1, vkDT::eStorageImage, &imageInfo};
    m_device.updateDescriptorSets(wds, nullptr);
  }
}

void Raytracer::createPipeline(const vk::DescriptorSetLayout& sceneDescSetLayout)
{
  std::vector<std::string> paths = defaultSearchPaths;
  vk::ShaderModule raygenSM = nvvk::createShaderModule(m_device, nvh::loadFile("shaders/raytrace.rgen.spv", true, paths));
  vk::ShaderModule missSM = nvvk::createShaderModule(m_device, nvh::loadFile("shaders/raytrace.rmiss.spv", true, paths));
  vk::ShaderModule shadowmissSM =
      nvvk::createShaderModule(m_device, nvh::loadFile("shaders/raytraceShadow.rmiss.spv", true, paths));
  vk::ShaderModule chitSM = nvvk::createShaderModule(m_device, nvh::loadFile("shaders/raytrace.rchit.spv", true, paths));

  std::vector<vk::PipelineShaderStageCreateInfo> stages;

  // Raygen
  vk::RayTracingShaderGroupCreateInfoNV rg{vk::RayTracingShaderGroupTypeNV::eGeneral, VK_SHADER_UNUSED_NV,
                                           VK_SHADER_UNUSED_NV, VK_SHADER_UNUSED_NV, VK_SHADER_UNUSED_NV};
  stages.push_back({{}, vk::ShaderStageFlagBits::eRaygenNV, raygenSM, "main"});
  rg.setGeneralShader(static_cast<uint32_t>(stages.size() - 1));
  m_groups.push_back(rg);
  // Miss
  vk::RayTracingShaderGroupCreateInfoNV mg{vk::RayTracingShaderGroupTypeNV::eGeneral, VK_SHADER_UNUSED_NV,
                                           VK_SHADER_UNUSED_NV, VK_SHADER_UNUSED_NV, VK_SHADER_UNUSED_NV};
  stages.push_back({{}, vk::ShaderStageFlagBits::eMissNV, missSM, "main"});
  mg.setGeneralShader(static_cast<uint32_t>(stages.size() - 1));
  m_groups.push_back(mg);
  // Shadow Miss
  stages.push_back({{}, vk::ShaderStageFlagBits::eMissNV, shadowmissSM, "main"});
  mg.setGeneralShader(static_cast<uint32_t>(stages.size() - 1));
  m_groups.push_back(mg);
  // Hit Group - Closest Hit + AnyHit
  vk::RayTracingShaderGroupCreateInfoNV hg{vk::RayTracingShaderGroupTypeNV::eTrianglesHitGroup, VK_SHADER_UNUSED_NV,
                                           VK_SHADER_UNUSED_NV, VK_SHADER_UNUSED_NV, VK_SHADER_UNUSED_NV};
  stages.push_back({{}, vk::ShaderStageFlagBits::eClosestHitNV, chitSM, "main"});
  hg.setClosestHitShader(static_cast<uint32_t>(stages.size() - 1));
  m_groups.push_back(hg);

  // Push constant: ray depth, ...
  vk::PushConstantRange pushConstant{vk::ShaderStageFlagBits::eRaygenNV | vk::ShaderStageFlagBits::eClosestHitNV
                                         | vk::ShaderStageFlagBits::eMissNV,
                                     0, sizeof(PushConstant)};

  // All 3 descriptors
  std::vector<vk::DescriptorSetLayout> allLayouts = {m_descSetLayout, sceneDescSetLayout};
  vk::PipelineLayoutCreateInfo         pipelineLayoutCreateInfo;
  pipelineLayoutCreateInfo.setSetLayoutCount(static_cast<uint32_t>(allLayouts.size()));
  pipelineLayoutCreateInfo.setPSetLayouts(allLayouts.data());
  pipelineLayoutCreateInfo.setPushConstantRangeCount(1);
  pipelineLayoutCreateInfo.setPPushConstantRanges(&pushConstant);
  m_pipelineLayout = m_device.createPipelineLayout(pipelineLayoutCreateInfo);
  m_debug.setObjectName(m_pipelineLayout, "raytracer");

  // Assemble the shader stages and recursion depth info into the raytracing pipeline
  vk::RayTracingPipelineCreateInfoNV rayPipelineInfo;
  rayPipelineInfo.setStageCount(static_cast<uint32_t>(stages.size()));
  rayPipelineInfo.setPStages(stages.data());
  rayPipelineInfo.setGroupCount(static_cast<uint32_t>(m_groups.size()));
  rayPipelineInfo.setPGroups(m_groups.data());
  rayPipelineInfo.setMaxRecursionDepth(10);
  rayPipelineInfo.setLayout(m_pipelineLayout);
  m_pipeline = static_cast<const vk::Pipeline&>(m_device.createRayTracingPipelineNV({}, rayPipelineInfo));

  m_device.destroyShaderModule(raygenSM);
  m_device.destroyShaderModule(missSM);
  m_device.destroyShaderModule(shadowmissSM);
  m_device.destroyShaderModule(chitSM);
}

//--------------------------------------------------------------------------------------------------
//
//
void Raytracer::createShadingBindingTable()
{
  auto     groupCount      = static_cast<uint32_t>(m_groups.size());   // 3 shaders: raygen, miss, chit
  uint32_t groupHandleSize = m_rtProperties.shaderGroupHandleSize;     // Size of a program identifier
  uint32_t baseAlignment   = m_rtProperties.shaderGroupBaseAlignment;  // Size of a program identifier


  // Fetch all the shader handles used in the pipeline, so that they can be written in the SBT
  uint32_t             sbtSize = groupCount * baseAlignment;
  std::vector<uint8_t> shaderHandleStorage(sbtSize);
  m_device.getRayTracingShaderGroupHandlesNV(m_pipeline, 0, groupCount, sbtSize, shaderHandleStorage.data());

  m_sbtBuffer = m_alloc->createBuffer(sbtSize, vk::BufferUsageFlagBits::eTransferSrc,
                                      vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
  m_debug.setObjectName(m_sbtBuffer.buffer, std::string("SBT").c_str());

  // Write the handles in the SBT
  void* mapped = m_alloc->map(m_sbtBuffer);
  auto* pData  = reinterpret_cast<uint8_t*>(mapped);
  for(uint32_t g = 0; g < groupCount; g++)
  {
    memcpy(pData, shaderHandleStorage.data() + g * groupHandleSize, groupHandleSize);  // raygen
    pData += baseAlignment;
  }
  m_alloc->unmap(m_sbtBuffer);
}

//--------------------------------------------------------------------------------------------------
//
//
void Raytracer::run(const vk::CommandBuffer& cmdBuf, const vk::DescriptorSet& sceneDescSet, int frame /*= 0*/)
{
  m_pushC.frame = frame;

  uint32_t progSize = m_rtProperties.shaderGroupBaseAlignment;  // Size of a program identifier
  cmdBuf.bindPipeline(vk::PipelineBindPoint::eRayTracingNV, m_pipeline);
  cmdBuf.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingNV, m_pipelineLayout, 0, {m_descSet, sceneDescSet}, {});
  cmdBuf.pushConstants<PushConstant>(m_pipelineLayout,
                                     vk::ShaderStageFlagBits::eRaygenNV | vk::ShaderStageFlagBits::eClosestHitNV
                                         | vk::ShaderStageFlagBits::eMissNV,
                                     0, m_pushC);

  vk::DeviceSize rayGenOffset   = 0 * progSize;
  vk::DeviceSize missOffset     = 1 * progSize;
  vk::DeviceSize missStride     = progSize;
  vk::DeviceSize hitGroupOffset = 3 * progSize;  // Jump over the 2 miss
  vk::DeviceSize hitGroupStride = progSize;

  cmdBuf.traceRaysNV(m_sbtBuffer.buffer, rayGenOffset,                    //
                     m_sbtBuffer.buffer, missOffset, missStride,          //
                     m_sbtBuffer.buffer, hitGroupOffset, hitGroupStride,  //
                     m_sbtBuffer.buffer, 0, 0,                            //
                     m_outputSize.width, m_outputSize.height,             //
                     1 /*, NVVKPP_DISPATCHER*/);
}

bool Raytracer::uiSetup()
{
  bool modified = false;
  if(ImGui::CollapsingHeader("Ray Tracing"))
  {
    modified = false;
    modified |= ImGui::SliderFloat("Max Ray Length", &m_pushC.maxRayLenght, 1, 1000000, "%.1f", 2.f);
    modified |= ImGui::SliderInt("Samples Per Frame", &m_pushC.samples, 1, 100);
    modified |= ImGui::SliderInt("Max Iteration ", &m_maxFrames, 1, 1000);
  }
  return modified;
}

void Raytracer::setClearColor(nvmath::vec3f& _color)
{
  m_pushC.backgroundColor = _color;
}

void Raytracer::setToonSteps(int nbStep)
{
  m_pushC.nbSteps = nbStep;
}

void Raytracer::setToonLightDir(nvmath::vec3f lightDir)
{
  m_pushC.lightDir = lightDir;
}
