#pragma once

#include <vulkan/vulkan.hpp>

#include "vk_util.hpp"

#include "nvh/fileoperations.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/shaders_vk.hpp"


extern std::vector<std::string> defaultSearchPaths;


// Find the min/max value of the depth value stored in the output data buffer
class CompDepthMinMax
{
public:
  void setup(const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t queueIndex, nvvk::Allocator* allocator)
  {
    m_device     = device;
    m_queueIndex = queueIndex;
    m_alloc      = allocator;
    m_debug.setup(device);

    // Create the buffer, no value in
    m_values = m_alloc->createBuffer(2 * sizeof(uint32_t), vkBU::eTransferSrc | vkBU::eTransferDst | vkBU::eStorageBuffer,
                                     vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    createCompDescriptors();
    createCompPipelines();
  }

  void setInput(const nvvk::Texture& dataImage)
  {
    std::vector<vk::WriteDescriptorSet> writes;
    writes.emplace_back(m_descSetBind.makeWrite(m_descSet, 0, &dataImage.descriptor));  // ray tracing
    vk::DescriptorBufferInfo bi{m_values.buffer, 0, VK_WHOLE_SIZE};
    writes.emplace_back(m_descSetBind.makeWrite(m_descSet, 1, &bi));  // ray tracing
    m_device.updateDescriptorSets(writes, nullptr);
  }

  void execute(const vk::CommandBuffer& cmdBuf, const vk::Extent2D& size)
  {
    float big{10000.f};
    m_minmax = {*reinterpret_cast<uint32_t*>(&big), 0};  // Resetting zNear and zFar values (floatBitsToInt)
    cmdBuf.updateBuffer<uint32_t>(m_values.buffer, 0, m_minmax);
    cmdBuf.bindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline);
    cmdBuf.bindDescriptorSets(vk::PipelineBindPoint::eCompute, m_pipelineLayout, 0, {m_descSet}, {});
    cmdBuf.dispatch(size.width / 32 + 1, size.height / 32 + 1, 1);

    // Adding a barrier to make sure the compute is done before one of the fragment
	// shader picks up the values computed here.
    vk::BufferMemoryBarrier bmb;
    bmb.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite);
    bmb.setDstAccessMask(vk::AccessFlagBits::eShaderRead);
    bmb.setOffset(0);
    bmb.setSize(VK_WHOLE_SIZE);
    bmb.setSize(VK_WHOLE_SIZE);
    bmb.setBuffer(m_values.buffer);
    cmdBuf.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eFragmentShader,
                           vk::DependencyFlagBits::eDeviceGroup, 0, nullptr, 1, &bmb, 0, nullptr);
  }

  void getNearFar(float& near_, float& far_)
  {
    {
      m_device.waitIdle();  // Must wait to be sure to get synchronization (use only for debug)
      void* mapped = m_alloc->map(m_values);
      memcpy(m_minmax.data(), mapped, 2 * sizeof(float));
      m_alloc->unmap(m_values);
    }

    near_ = *reinterpret_cast<float*>(&m_minmax[0]);  // intBitsToFloat
    far_  = *reinterpret_cast<float*>(&m_minmax[1]);
  }
  const nvvk::Buffer& getBuffer() { return m_values; }  // zNear - zFar

  void destroy()
  {
    m_alloc->destroy(m_values);
    m_device.destroyDescriptorSetLayout(m_descSetLayout);
    m_device.destroyPipeline(m_pipeline);
    m_device.destroyPipelineLayout(m_pipelineLayout);
    m_device.destroyDescriptorPool(m_descPool);
  }


private:
  void createCompDescriptors()
  {
    m_descSetBind.addBinding(vkDS(0, vkDT::eStorageImage, 1, vkSS::eCompute));   // Input - image
    m_descSetBind.addBinding(vkDS(1, vkDT::eStorageBuffer, 1, vkSS::eCompute));  // Output - values

    m_descSetLayout = m_descSetBind.createLayout(m_device);
    m_descPool      = m_descSetBind.createPool(m_device, 1);
    m_descSet       = nvvk::allocateDescriptorSet(m_device, m_descPool, m_descSetLayout);
  }

  void createCompPipelines()
  {
    vk::PipelineLayoutCreateInfo layout_info{{}, 1, &m_descSetLayout};
    m_pipelineLayout = m_device.createPipelineLayout(layout_info);
    m_debug.setObjectName(m_pipelineLayout, "minmax");
    vk::ComputePipelineCreateInfo createInfo{{}, {}, m_pipelineLayout};

    createInfo.stage = nvvk::createShaderStageInfo(m_device, nvh::loadFile("shaders/depthminmax.comp.spv", true, defaultSearchPaths),
                                                   VK_SHADER_STAGE_COMPUTE_BIT);
    m_pipeline = static_cast<const vk::Pipeline&>(m_device.createComputePipeline({}, createInfo, nullptr));
    m_device.destroy(createInfo.stage.module);
  }

  vk::Device       m_device;
  uint32_t         m_queueIndex;
  nvvk::Allocator* m_alloc{nullptr};
  nvvk::DebugUtil  m_debug;

  std::array<uint32_t, 2> m_minmax;
  nvvk::Buffer            m_values;  // min/max

  nvvk::DescriptorSetBindings m_descSetBind;
  vk::DescriptorPool          m_descPool;
  vk::DescriptorSetLayout     m_descSetLayout;
  vk::DescriptorSet           m_descSet;
  vk::Pipeline                m_pipeline;
  vk::PipelineLayout          m_pipelineLayout;
};
