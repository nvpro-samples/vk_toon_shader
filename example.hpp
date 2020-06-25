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

#include <vulkan/vulkan.hpp>

#include "vk_util.hpp"

#include <array>
#include <nvmath/nvmath.h>

#include "comp_depth_minmax.hpp"
#include "nvh/gltfscene.hpp"
#include "nvvk/appbase_vkpp.hpp"
#include "nvvk/gizmos_vk.hpp"
#include "nvvk/images_vk.hpp"
#include "post_compositing.hpp"
#include "post_kuwahara.hpp"
#include "post_kuwahara_aniso.hpp"
#include "post_nrmdepth.hpp"
#include "post_objcontour.hpp"
#include "post_tonemapper.hpp"
#include "rasterizer.hpp"
#include "raypick.hpp"
#include "raytracer.hpp"


//--------------------------------------------------------------------------------------------------
// Loading a glTF scene, raytrace and tonemap result
//
class VkToonExample : public nvvk::AppBase
{
public:
  VkToonExample() = default;

  void loadScene(const std::string& filename);
  void createPostProcess();
  void display();

  void setup(const vk::Instance& instance, const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t graphicsQueueIndex) override;

  void destroy() override;
  void onResize(int w, int h) override;
  void createRenderPass() override;
  void createAxis();
  void createDescriptorFinal();
  void createFinalPipeline();
  void onKeyboard(int key, int scancode, int action, int mods) override;
  void onFileDrop(const char* filename) override;

private:
  struct SceneUBO
  {
    nvmath::mat4f projection;
    nvmath::mat4f model;
    nvmath::vec4f cameraPosition{0.f, 0.f, 0.f};
  };

  void createDescriptorMaterial();
  void createDescriptorRaytrace();
  void createPipeline();
  void createSceneBuffers();
  void createSceneDescriptors();
  void drawUI();
  void importImages(tinygltf::Model& gltfModel);
  void prepareUniformBuffers();
  void resetFrame();
  void settingPostPipeline();
  void updateCameraBuffer(const vk::CommandBuffer& cmdBuffer);
  void updateDescriptor(const vk::DescriptorImageInfo& descriptor);
  void updateFrame();

  vk::GeometryNV primitiveToGeometry(const nvh::GltfPrimMesh& prim);


  vk::RenderPass     m_renderPassUI;
  vk::PipelineLayout m_pipelineLayout;
  vk::Pipeline       m_pipeline;

  // Descriptors
  enum Dset
  {
    eFinal,  // For the tonemapper
    eScene,  // All scene data
    Total
  };
  std::vector<vk::DescriptorSetLayout>     m_descSetLayout{Dset::Total};
  std::vector<vk::DescriptorPool>          m_descPool{Dset::Total};
  std::vector<vk::DescriptorSet>           m_descSet{Dset::Total};
  std::vector<nvvk::DescriptorSetBindings> m_descSetLayoutBind{Dset::Total};

  // GLTF scene model
  nvh::GltfScene m_gltfScene;   // The scene
  nvh::GltfStats m_sceneStats;  // The scene stats
  SceneUBO       m_sceneUbo;    // Camera, light and more

  int m_frameNumber{0};

  nvvk::AxisVK m_axis;        // To display the axis in the lower left corner
  Raytracer    m_raytracer;   // The raytracer
  RayPicker    m_rayPicker;   // Picking under mouse using raytracer
  Rasterizer   m_rasterizer;  // Rasterizer

  // Post effect
  Tonemapper        m_tonemapper;  //
  PostNrmDepth      m_nrmDepth;
  PostObjContour    m_objContour;
  PostCompositing   m_compositing;
  PostKuwahara      m_kuwahara;
  PostKuwaharaAniso m_kuwaharaAniso;
  CompDepthMinMax   m_depthMinMax;

  // All buffers on the Device
  nvvk::Buffer m_sceneBuffer;
  nvvk::Buffer m_vertexBuffer;
  nvvk::Buffer m_normalBuffer;
  nvvk::Buffer m_uvBuffer;
  nvvk::Buffer m_indexBuffer;
  nvvk::Buffer m_materialBuffer;
  nvvk::Buffer m_matrixBuffer;

  // All textures
  std::vector<nvvk::Texture> m_textures;

  bool          m_useRaytracer{true};
  nvmath::vec3f m_backgroundColor{1.f};      // clear color and miss
  int           m_toonNbStep{5};             // Toon shading steps
  nvmath::vec3f m_toonLightDir{-1, -1, -1};  // Toon light

  // Memory allocator for buffers and images
  nvvk::DeviceMemoryAllocator m_dmaAllocator;
  nvvk::AllocatorDma          m_alloc;

  nvvk::DebugUtil m_debug;
};
