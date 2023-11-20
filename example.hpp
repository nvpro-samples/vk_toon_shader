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


#include <vulkan/vulkan.hpp>

#include "vk_util.hpp"

#include <array>

#include "comp_depth_minmax.hpp"
#include "nvh/gltfscene.hpp"
#include "nvvkhl/appbase_vkpp.hpp"
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
class VkToonExample : public nvvkhl::AppBase
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
    glm::mat4 projection;
    glm::mat4 model;
    glm::vec4 cameraPosition{0.f, 0.f, 0.f, 1.f};
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
  std::vector<vk::DescriptorSetLayout>       m_descSetLayout{Dset::Total};
  std::vector<vk::DescriptorPool>            m_descPool{Dset::Total};
  std::vector<vk::DescriptorSet>             m_descSet{Dset::Total};
  std::vector<nvvkpp::DescriptorSetBindings> m_descSetLayoutBind{Dset::Total};

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

  bool      m_useRaytracer{true};
  glm::vec3 m_backgroundColor{1.f};      // clear color and miss
  int       m_toonNbStep{5};             // Toon shading steps
  glm::vec3 m_toonLightDir{-1, -1, -1};  // Toon light

  // Allocator for buffers and images
  Allocator m_alloc;

  nvvk::DebugUtil m_debug;
};
