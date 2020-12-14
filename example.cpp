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
// This example is loading a glTF scene and raytrace it with a very simple material
//
#include <iostream>

#include <filesystem>
#include <vulkan/vulkan.hpp>

#include "example.hpp"
#include "imgui/imgui_orient.h"
#include "imgui_impl_glfw.h"
#include "imgui_internal.h"
#include "nvh/fileoperations.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/profiler_vk.hpp"
#include "nvvk/renderpasses_vk.hpp"
#include "raytracer.hpp"
#include "shaders/binding.glsl"
#include "shaders/gltf.glsl"

// Define these only in *one* .cc file.
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <fileformats/tiny_gltf.h>


extern std::vector<std::string> defaultSearchPaths;


void VkToonExample::setup(const vk::Instance& instance, const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t graphicsQueueIndex)
{
  AppBase::setup(instance, device, physicalDevice, graphicsQueueIndex);
  m_debug.setup(device);

  m_dmaAllocator.init(device, physicalDevice);
  m_alloc.init(device, physicalDevice, &m_dmaAllocator);

  m_raytracer.setup(device, physicalDevice, graphicsQueueIndex, &m_alloc);
  m_rayPicker.setup(device, physicalDevice, graphicsQueueIndex, &m_alloc);
  m_rasterizer.setup(device, physicalDevice, graphicsQueueIndex, &m_alloc);

  m_tonemapper.setup(device, physicalDevice, graphicsQueueIndex, &m_alloc);
  m_nrmDepth.setup(device, physicalDevice, graphicsQueueIndex, &m_alloc);
  m_objContour.setup(device, physicalDevice, graphicsQueueIndex, &m_alloc);
  m_compositing.setup(device, physicalDevice, graphicsQueueIndex, &m_alloc);
  m_kuwahara.setup(device, physicalDevice, graphicsQueueIndex, &m_alloc);
  m_kuwaharaAniso.setup(device, physicalDevice, graphicsQueueIndex, &m_alloc);
  m_depthMinMax.setup(device, physicalDevice, graphicsQueueIndex, &m_alloc);
}


//--------------------------------------------------------------------------------------------------
// Overridden function that is called after the base class create()
//
void VkToonExample::loadScene(const std::string& filename)
{
  // Loading the glTF file, it will allocate 3 buffers: vertex, index and matrices
  tinygltf::Model    tmodel;
  tinygltf::TinyGLTF tcontext;
  std::string        warn, error;
  bool               fileLoaded = false;
  MilliTimer         timer;

  // Loading glTF file
  {
    LOGI("Loading glTF: %s\n", filename.c_str());
    fileLoaded = tcontext.LoadASCIIFromFile(&tmodel, &error, &warn, filename);
    if(!warn.empty())
      LOGE("Warning loading %s: %s", filename.c_str(), warn.c_str());
    if(!error.empty())
      LOGE("Error loading %s: %s", filename.c_str(), error.c_str());
    assert(fileLoaded && error.empty() && error.c_str());
    LOGI(" --> (%5.3f ms)\n", timer.elapse());
  }

  // From tinyGLTF to our glTF representation
  {
    LOGI("Importing Scene\n");
    m_gltfScene.importMaterials(tmodel);
    m_gltfScene.importDrawableNodes(tmodel, nvh::GltfAttributes::Normal | nvh::GltfAttributes::Texcoord_0);
    m_sceneStats = m_gltfScene.getStatistics(tmodel);
    LOGI(" --> (%5.3f ms)\n", timer.elapse());
  }

  // Uploading images on GPU
  importImages(tmodel);

  // Set the camera to see the scene
  if(!m_gltfScene.m_cameras.empty())
  {
    CameraManip.setLookat(m_gltfScene.m_cameras[0].eye, m_gltfScene.m_cameras[0].center, m_gltfScene.m_cameras[0].up);
  }

  // Create buffers with all scene information: vertex, normal, material, ...
  createSceneBuffers();
  createSceneDescriptors();

  // Converting the scene to ray tracing
  if(m_raytracer.isValid())
  {
    LOGI("Creating BLAS and TLAS\n");

    // BLAS - Storing each primitive in a geometry
    std::vector<std::vector<VkGeometryNV>> blass;
    std::vector<RtPrimitiveLookup>         primLookup;
    for(auto& primMesh : m_gltfScene.m_primMeshes)
    {
      auto geo = primitiveToGeometry(primMesh);
      blass.push_back({geo});

      // The following is use to find the primitive mesh information in the CHIT
      primLookup.push_back({primMesh.firstIndex, primMesh.vertexOffset, primMesh.materialIndex});
    }
    m_raytracer.builder().buildBlas(blass);
    m_raytracer.setPrimitiveLookup(primLookup);

    // TLAS - Top level for each valid mesh
    std::vector<nvvk::RaytracingBuilderNV::Instance> rayInst;
    for(auto& node : m_gltfScene.m_nodes)
    {
      nvvk::RaytracingBuilderNV::Instance inst;
      inst.transform  = node.worldMatrix;
      inst.instanceId = node.primMesh;  // gl_InstanceCustomIndexNV
      inst.blasId     = node.primMesh;
      rayInst.emplace_back(inst);

      auto& mesh = m_gltfScene.m_primMeshes[node.primMesh];
    }
    m_raytracer.builder().buildTlas(rayInst);

    // Raytracing
    m_raytracer.createOutputImages(m_size);
    m_raytracer.createDescriptorSet();
    m_raytracer.createPipeline(m_descSetLayout[eScene]);
    m_raytracer.createShadingBindingTable();

    // Raytracer Picker : Using -SPACE- to pick an object
    // NOTE: using ray-tracer so if no ray-tracer, no ray-picker
    vk::DescriptorBufferInfo sceneDesc{m_sceneBuffer.buffer, 0, VK_WHOLE_SIZE};
    m_rayPicker.initialize(m_raytracer.builder().getAccelerationStructure(), sceneDesc);
  }

  // Rasterizer
  m_rasterizer.setObjectPointers(&m_gltfScene, &m_vertexBuffer, &m_normalBuffer, &m_uvBuffer, &m_indexBuffer);
  m_rasterizer.createRenderPass();
  m_rasterizer.createOutputImages(m_size);
  m_rasterizer.createPipeline(m_descSetLayout[eScene]);
  m_rasterizer.recordCommandBuffer(m_cmdPool, m_descSet[eScene]);
}

//--------------------------------------------------------------------------------------------------
// Creating all elements of the post-process: line extractions, compositing, ..
//
void VkToonExample::createPostProcess()
{
  // Post-process tonemapper
  m_tonemapper.initialize(m_size);
  m_nrmDepth.initialize(m_size);
  m_objContour.initialize(m_size);
  m_compositing.initialize(m_size);
  m_kuwahara.initialize(m_size);
  m_kuwaharaAniso.initialize(m_size);
  // Setting the post effect pipeline
  settingPostPipeline();
}


//--------------------------------------------------------------------------------------------------
// Converting a GLTF primitive to VkGeometryNV used to create a BLAS
//
vk::GeometryNV VkToonExample::primitiveToGeometry(const nvh::GltfPrimMesh& prim)
{
  vk::GeometryTrianglesNV triangles;
  triangles.setVertexData(m_vertexBuffer.buffer);
  triangles.setVertexOffset(prim.vertexOffset * sizeof(nvmath::vec3f));
  triangles.setVertexCount(prim.vertexCount);
  triangles.setVertexStride(sizeof(nvmath::vec3f));
  triangles.setVertexFormat(vk::Format::eR32G32B32Sfloat);
  triangles.setIndexData(m_indexBuffer.buffer);
  triangles.setIndexOffset(prim.firstIndex * sizeof(uint32_t));
  triangles.setIndexCount(prim.indexCount);
  triangles.setIndexType(vk::IndexType::eUint32);  // 32-bit indices
  vk::GeometryDataNV geoData;
  geoData.setTriangles(triangles);
  vk::GeometryNV geometry;
  geometry.setGeometry(geoData);
  geometry.setFlags(vk::GeometryFlagBitsNV::eNoDuplicateAnyHitInvocation);
  return geometry;
}


//--------------------------------------------------------------------------------------------------
// Overridden function called on shutdown
//
void VkToonExample::destroy()
{
  m_device.waitIdle();

  m_gltfScene.destroy();
  m_alloc.destroy(m_vertexBuffer);
  m_alloc.destroy(m_normalBuffer);
  m_alloc.destroy(m_uvBuffer);
  m_alloc.destroy(m_indexBuffer);
  m_alloc.destroy(m_materialBuffer);
  m_alloc.destroy(m_matrixBuffer);
  m_alloc.destroy(m_sceneBuffer);

  for(auto& t : m_textures)
  {
    m_alloc.destroy(t);
  }

  m_device.destroyRenderPass(m_renderPassUI);
  m_renderPassUI = vk::RenderPass();

  m_device.destroyPipeline(m_pipeline);
  m_device.destroyPipelineLayout(m_pipelineLayout);
  for(int i = 0; i < Dset::Total; i++)
  {
    m_device.destroyDescriptorSetLayout(m_descSetLayout[i]);
    m_device.destroyDescriptorPool(m_descPool[i]);
  }

  m_rayPicker.destroy();
  m_axis.deinit();

  m_tonemapper.destroy();
  m_nrmDepth.destroy();
  m_objContour.destroy();
  m_compositing.destroy();
  m_kuwahara.destroy();
  m_kuwaharaAniso.destroy();
  m_depthMinMax.destroy();

  m_raytracer.destroy();
  m_rasterizer.destroy();

  m_alloc.deinit();
  m_dmaAllocator.deinit();

  AppBase::destroy();
}

//--------------------------------------------------------------------------------
// Called at each frame, as fast as possible
//
void VkToonExample::display()
{
  updateFrame();

  drawUI();

  // render the scene
  prepareFrame();
  const vk::CommandBuffer& cmdBuf = m_commandBuffers[getCurFrame()];
  cmdBuf.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

  // Updating the matrices of the camera
  updateCameraBuffer(cmdBuf);

  std::array<vk::ClearValue, 2> clearValues;
  clearValues[0].setColor(makeClearColor(m_backgroundColor));
  clearValues[1].setDepthStencil({1.0f, 0});

  {
    // Raytracing
    if(m_raytracer.isValid() && m_useRaytracer)
    {
      if((m_frameNumber < m_raytracer.maxFrames()))
      {
        auto dgbLabel = m_debug.scopeLabel(cmdBuf, "raytracing");
        m_raytracer.run(cmdBuf, m_descSet[eScene], m_frameNumber);

        // Extract the near and far, and set those values to the Normal/Depth extraction
        m_depthMinMax.execute(cmdBuf, m_size);
      }
    }
    else  // Rasterizing
    {
      auto dgbLabel = m_debug.scopeLabel(cmdBuf, "rasterizing");
      m_rasterizer.run(cmdBuf, m_descSet[eScene], m_frameNumber);

      // Extract the near and far, and set those values to the Normal/Depth extraction
      m_depthMinMax.execute(cmdBuf, m_size);
    }

    // Apply tonemapper, its output is set in the descriptor set
    {
      auto dgbLabel = m_debug.scopeLabel(cmdBuf, "tonemapping");
      m_tonemapper.execute(cmdBuf);
    }

    // Apply toonshader, its output is set in the descriptor set
    {
      auto dgbLabel = m_debug.scopeLabel(cmdBuf, "toonshader");

      m_nrmDepth.execute(cmdBuf);
      m_objContour.execute(cmdBuf);
      if(m_kuwahara.isActive())
      {
        if(m_kuwaharaAniso.isActive())
          m_kuwaharaAniso.execute(cmdBuf);
        else
          m_kuwahara.execute(cmdBuf);
      }

      m_compositing.execute(cmdBuf);
    }


    // Drawing a quad (pass through + final.frag)
    {
      auto dgbLabel = m_debug.scopeLabel(cmdBuf, "display");

      vk::RenderPassBeginInfo renderPassBeginInfo = {
          m_renderPass, m_framebuffers[getCurFrame()], {{}, m_size}, 2, clearValues.data()};
      cmdBuf.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

      cmdBuf.setViewport(0, {vk::Viewport(0.0f, 0.0f, static_cast<float>(m_size.width), static_cast<float>(m_size.height), 0.0f, 1.0f)});
      cmdBuf.setScissor(0, {{{0, 0}, {m_size.width, m_size.height}}});

      //setViewport(cmdBuf);

      cmdBuf.bindPipeline(vk::PipelineBindPoint::eGraphics, m_pipeline);
      cmdBuf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_pipelineLayout, 0, m_descSet[Dset::eFinal], {});

      cmdBuf.draw(3, 1, 0, 0);
    }
    cmdBuf.endRenderPass();

    vk::RenderPassBeginInfo renderPassBeginInfo = {
        m_renderPassUI, m_framebuffers[getCurFrame()], {{}, m_size}, 2, clearValues.data()};
    cmdBuf.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);
    // Rendering axis in same render pass
    {
      auto dgbLabel = m_debug.scopeLabel(cmdBuf, "Axes");
      m_axis.display(cmdBuf, CameraManip.getMatrix(), m_size);
    }

    {
      auto dgbLabel = m_debug.scopeLabel(cmdBuf, "ImGui");

      // Drawing GUI
      ImGui::Render();
      ImDrawData* imguiDrawData = ImGui::GetDrawData();
      ImGui::RenderDrawDataVK(cmdBuf, imguiDrawData);
      ImGui::EndFrame();
    }


    cmdBuf.endRenderPass();
  }


  // End of the frame and present the one which is ready
  cmdBuf.end();
  submitFrame();
}


//--------------------------------------------------------------------------------------------------
// Return the current frame number
// Check if the camera matrix has changed, if yes, then reset the frame to 0
// otherwise, increment
//
void VkToonExample::updateFrame()
{
  static nvmath::mat4f refCamMatrix;
  static float         fov = 0;

  auto& m = CameraManip.getMatrix();
  if(memcmp(&refCamMatrix.a00, &m.a00, sizeof(nvmath::mat4f)) != 0 || fov != CameraManip.getFov())
  {
    resetFrame();
    refCamMatrix = m;
    fov          = CameraManip.getFov();
  }
  m_frameNumber++;
}

void VkToonExample::resetFrame()
{
  m_frameNumber = -1;
}

//--------------------------------------------------------------------------------------------------
// Creating the Uniform Buffers, only for the scene camera matrices
// The one holding all all matrices of the scene nodes was created in glTF.load()
//
void VkToonExample::createSceneBuffers()
{
  {
    nvvk::ScopeCommandBuffer cmdBuf(m_device, m_graphicsQueueIndex);
    m_sceneBuffer = m_alloc.createBuffer(cmdBuf, sizeof(SceneUBO), nullptr, vkBU::eUniformBuffer);

    // Creating the GPU buffer of the vertices
    m_vertexBuffer = m_alloc.createBuffer(cmdBuf, m_gltfScene.m_positions, vkBU::eVertexBuffer | vkBU::eStorageBuffer);
    m_normalBuffer = m_alloc.createBuffer(cmdBuf, m_gltfScene.m_normals, vkBU::eVertexBuffer | vkBU::eStorageBuffer);
    m_uvBuffer     = m_alloc.createBuffer(cmdBuf, m_gltfScene.m_texcoords0, vkBU::eVertexBuffer | vkBU::eStorageBuffer);
    m_indexBuffer  = m_alloc.createBuffer(cmdBuf, m_gltfScene.m_indices, vkBU::eIndexBuffer | vkBU::eStorageBuffer);


    std::vector<GltfShadeMaterial> shadeMaterials;
    for(auto& m : m_gltfScene.m_materials)
    {
      shadeMaterials.emplace_back(GltfShadeMaterial{m.shadingModel,
                                                    m.pbrBaseColorFactor,
                                                    m.pbrBaseColorTexture,
                                                    m.pbrMetallicFactor,
                                                    m.pbrRoughnessFactor,
                                                    m.pbrMetallicRoughnessTexture,
                                                    m.khrDiffuseFactor,
                                                    m.khrDiffuseTexture,
                                                    m.khrSpecularFactor,
                                                    m.khrGlossinessFactor,
                                                    m.khrSpecularGlossinessTexture,
                                                    m.emissiveTexture,
                                                    m.emissiveFactor,
                                                    m.alphaMode,
                                                    m.alphaCutoff,
                                                    m.doubleSided,
                                                    m.normalTexture,
                                                    m.normalTextureScale,
                                                    m.occlusionTexture,
                                                    m.occlusionTextureStrength

      });
    }
    m_materialBuffer = m_alloc.createBuffer(cmdBuf, shadeMaterials, vkBU::eStorageBuffer);


    // Instance Matrices used by rasterizer
    struct sInstMat
    {
      mat4 matrix;
      mat4 matrixIT;
    };
    std::vector<sInstMat> nodeMatrices;
    for(auto& node : m_gltfScene.m_nodes)
    {
      sInstMat mat;
      mat.matrix   = node.worldMatrix;
      mat.matrixIT = nvmath::transpose(nvmath::invert(node.worldMatrix));
      nodeMatrices.emplace_back(mat);
    }
    m_matrixBuffer = m_alloc.createBuffer(cmdBuf, nodeMatrices, vkBU::eStorageBuffer);
  }
  m_alloc.finalizeAndReleaseStaging();

  m_debug.setObjectName(m_sceneBuffer.buffer, "SceneUbo");
  m_debug.setObjectName(m_vertexBuffer.buffer, "Vertex");
  m_debug.setObjectName(m_indexBuffer.buffer, "Index");
  m_debug.setObjectName(m_normalBuffer.buffer, "Normal");
  m_debug.setObjectName(m_uvBuffer.buffer, "TexCoord");
  m_debug.setObjectName(m_materialBuffer.buffer, "Material");
  m_debug.setObjectName(m_matrixBuffer.buffer, "Matrix");
}

//--------------------------------------------------------------------------------------------------
// The pipeline is how things are rendered, which shaders, type of primitives, depth test and more
// This one is for displaying the ray traced image on a quad
//
void VkToonExample::createFinalPipeline()
{
  std::vector<std::string> paths = defaultSearchPaths;

  // Creating the pipeline layout
  vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;
  pipelineLayoutCreateInfo.setSetLayoutCount(1);
  pipelineLayoutCreateInfo.setPSetLayouts(&m_descSetLayout[eFinal]);
  m_pipelineLayout = m_device.createPipelineLayout(pipelineLayoutCreateInfo);
  m_debug.setObjectName(m_pipelineLayout, "Final");

  // Pipeline: completely generic, no vertices
  nvvk::GraphicsPipelineGeneratorCombined pipelineGenerator(m_device, m_pipelineLayout, m_renderPass);
  pipelineGenerator.addShader(nvh::loadFile("shaders/passthrough.vert.spv", true, paths), vk::ShaderStageFlagBits::eVertex);
  pipelineGenerator.addShader(nvh::loadFile("shaders/final.frag.spv", true, paths), vk::ShaderStageFlagBits::eFragment);
  pipelineGenerator.rasterizationState.setCullMode(vk::CullModeFlagBits::eNone);

  m_pipeline = pipelineGenerator.createPipeline();
}


//--------------------------------------------------------------------------------------------------
// The descriptor layout is the description of the data that is passed to the vertex or the
// fragment program.
//
void VkToonExample::createDescriptorFinal()
{
  m_descSetLayoutBind[eFinal].clear();
  m_descSetLayoutBind[eFinal].addBinding(vkDS(0, vkDT::eCombinedImageSampler, 1, vkSS::eFragment));
  m_descSetLayout[eFinal] = m_descSetLayoutBind[eFinal].createLayout(m_device);
  m_descPool[eFinal]      = m_descSetLayoutBind[eFinal].createPool(m_device);
  m_descSet[eFinal]       = nvvk::allocateDescriptorSet(m_device, m_descPool[eFinal], m_descSetLayout[eFinal]);
}

//--------------------------------------------------------------------------------------------------
// Creates all descriptors for raytracing (set 1)
//
void VkToonExample::createSceneDescriptors()
{
  using vkDSLB = vk::DescriptorSetLayoutBinding;
  m_descSetLayoutBind[eScene].clear();

  auto& bind = m_descSetLayoutBind[eScene];
  bind.addBinding(vkDSLB(B_SCENE, vkDT::eUniformBuffer, 1,
                         vkSS::eVertex | vkSS::eFragment | vkSS::eRaygenNV | vkSS::eClosestHitNV));     // Scene, camera
  bind.addBinding(vkDSLB(B_VERTICES, vkDT::eStorageBuffer, 1, vkSS::eClosestHitNV | vkSS::eAnyHitNV));  // Vertices
  bind.addBinding(vkDSLB(B_INDICES, vkDT::eStorageBuffer, 1, vkSS::eClosestHitNV | vkSS::eAnyHitNV));   // Indices
  bind.addBinding(vkDSLB(B_NORMALS, vkDT::eStorageBuffer, 1, vkSS::eClosestHitNV));                     // Normals
  bind.addBinding(vkDSLB(B_TEXCOORDS, vkDT::eStorageBuffer, 1, vkSS::eClosestHitNV));                   // TexCoord
  bind.addBinding(vkDSLB(B_MATERIAL, vkDT::eStorageBuffer, 1, vkSS::eFragment | vkSS::eClosestHitNV | vkSS::eAnyHitNV));  // material
  bind.addBinding(vkDSLB(B_MATRIX, vkDT::eStorageBuffer, 1,
                         vkSS::eVertex | vkSS::eClosestHitNV | vkSS::eAnyHitNV));  // matrix
  auto nbTextures = static_cast<uint32_t>(m_textures.size());
  bind.addBinding(vkDSLB(B_TEXTURES, vkDT::eCombinedImageSampler, nbTextures,
                         vkSS::eFragment | vkSS::eClosestHitNV | vkSS::eAnyHitNV));  // all textures

  m_descPool[eScene]      = m_descSetLayoutBind[eScene].createPool(m_device);
  m_descSetLayout[eScene] = m_descSetLayoutBind[eScene].createLayout(m_device);
  m_descSet[eScene]       = m_device.allocateDescriptorSets({m_descPool[eScene], 1, &m_descSetLayout[eScene]})[0];

  vk::DescriptorBufferInfo sceneDesc{m_sceneBuffer.buffer, 0, VK_WHOLE_SIZE};
  vk::DescriptorBufferInfo vertexDesc{m_vertexBuffer.buffer, 0, VK_WHOLE_SIZE};
  vk::DescriptorBufferInfo indexDesc{m_indexBuffer.buffer, 0, VK_WHOLE_SIZE};
  vk::DescriptorBufferInfo normalDesc{m_normalBuffer.buffer, 0, VK_WHOLE_SIZE};
  vk::DescriptorBufferInfo texcoordDesc{m_uvBuffer.buffer, 0, VK_WHOLE_SIZE};
  vk::DescriptorBufferInfo materialDesc{m_materialBuffer.buffer, 0, VK_WHOLE_SIZE};
  vk::DescriptorBufferInfo matrixDesc{m_matrixBuffer.buffer, 0, VK_WHOLE_SIZE};

  std::vector<vk::DescriptorImageInfo> dbiImages;
  for(const auto& imageDesc : m_textures)
    dbiImages.emplace_back(imageDesc.descriptor);

  std::vector<vk::WriteDescriptorSet> writes;
  writes.emplace_back(bind.makeWrite(m_descSet[eScene], B_SCENE, &sceneDesc));
  writes.emplace_back(bind.makeWrite(m_descSet[eScene], B_VERTICES, &vertexDesc));
  writes.emplace_back(bind.makeWrite(m_descSet[eScene], B_INDICES, &indexDesc));
  writes.emplace_back(bind.makeWrite(m_descSet[eScene], B_NORMALS, &normalDesc));
  writes.emplace_back(bind.makeWrite(m_descSet[eScene], B_TEXCOORDS, &texcoordDesc));
  writes.emplace_back(bind.makeWrite(m_descSet[eScene], B_MATERIAL, &materialDesc));
  writes.emplace_back(bind.makeWrite(m_descSet[eScene], B_MATRIX, &matrixDesc));

  for(int i = 0; i < dbiImages.size(); i++)
    writes.emplace_back(m_descSet[eScene], B_TEXTURES, i, 1, vk::DescriptorType::eCombinedImageSampler, &dbiImages[i]);

  m_device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}


//--------------------------------------------------------------------------------------------------
// Update the output
//
void VkToonExample::updateDescriptor(const vk::DescriptorImageInfo& descriptor)
{
  vk::WriteDescriptorSet writeDescriptorSets = m_descSetLayoutBind[eFinal].makeWrite(m_descSet[eFinal], 0, &descriptor);
  m_device.updateDescriptorSets(writeDescriptorSets, nullptr);
}

//--------------------------------------------------------------------------------------------------
// When the frames are redone, we also need to re-record the command buffer
//
void VkToonExample::onResize(int w, int h)
{
  if(m_raytracer.isValid())
  {
    m_raytracer.createOutputImages(m_size);
    m_raytracer.updateDescriptorSet();
  }
  m_rasterizer.createOutputImages(m_size);
  m_tonemapper.updateRenderTarget(m_size);

  m_nrmDepth.updateRenderTarget(m_size);
  m_objContour.updateRenderTarget(m_size);
  m_compositing.updateRenderTarget(m_size);
  m_kuwahara.updateRenderTarget(m_size);
  m_kuwaharaAniso.updateRenderTarget(m_size);

  settingPostPipeline();
  resetFrame();
}

void VkToonExample::createAxis()
{
  m_axis.init(m_device, m_renderPass, 0, 40.f);
}

//--------------------------------------------------------------------------------------------------
// Called at each frame to update the camera matrix
//
void VkToonExample::updateCameraBuffer(const vk::CommandBuffer& cmdBuffer)
{
  const float aspectRatio = m_size.width / static_cast<float>(m_size.height);
  float       nearPlane   = m_gltfScene.m_dimensions.radius / 10.0f;
  float       farPlane    = m_gltfScene.m_dimensions.radius * 50.0f;

  m_sceneUbo.model      = CameraManip.getMatrix();
  m_sceneUbo.projection = nvmath::perspectiveVK(CameraManip.getFov(), aspectRatio, nearPlane, farPlane);
  nvmath::vec3f pos, center, up;
  CameraManip.getLookat(pos, center, up);
  m_sceneUbo.cameraPosition = pos;

  cmdBuffer.updateBuffer<VkToonExample::SceneUBO>(m_sceneBuffer.buffer, 0, m_sceneUbo);
}

//--------------------------------------------------------------------------------------------------
// The display will render the recorded command buffer, then in a sub-pass, render the UI
//
void VkToonExample::createRenderPass()
{
  m_renderPass = nvvk::createRenderPass(m_device, {getColorFormat()}, m_depthFormat, 1, true, true);
  m_renderPassUI =
      nvvk::createRenderPass(m_device, {getColorFormat()}, m_depthFormat, 1, false, true, vk::ImageLayout::ePresentSrcKHR);
}

//--------------------------------------------------------------------------------------------------
// The post-pipeline, is sequence of post-process effects
// Tonemapper, normal-depth and object ID contour extraction and min/max depth value are all extracted from
// the result of the 2 images from rasterizer or raytracer.
//
// The Kuwahara effect is applied on the tonemapped image
//
// Finally, the composition of the image is the result of tonemapped/kuwahara(if activated) and the
// contour lines: normal/depth (details) and object id
//
void VkToonExample::settingPostPipeline()
{
  m_device.waitIdle();

  // Set all inputs
  if(m_raytracer.isValid() && m_useRaytracer)
  {
    m_tonemapper.setInputs(m_raytracer.outputImages());
    m_nrmDepth.setInputs({m_raytracer.outputImages()[1]}, m_depthMinMax.getBuffer());
    m_objContour.setInputs({m_raytracer.outputImages()[1]});
    m_depthMinMax.setInput(m_raytracer.outputImages()[1]);
  }
  else
  {
    m_tonemapper.setInputs(m_rasterizer.outputImages());
    m_nrmDepth.setInputs({m_rasterizer.outputImages()[1]}, m_depthMinMax.getBuffer());
    m_objContour.setInputs({m_rasterizer.outputImages()[1]});
    m_depthMinMax.setInput(m_rasterizer.outputImages()[1]);
  }

  m_kuwahara.setInputs({m_tonemapper.getOutput()});
  m_kuwaharaAniso.setInputs({m_tonemapper.getOutput()});


  // Compositing
  const nvvk::Texture& color = m_kuwahara.isActive() ?
                                   (m_kuwaharaAniso.isActive() ? m_kuwaharaAniso.getOutput() : m_kuwahara.getOutput()) :
                                   m_tonemapper.getOutput();
  const nvvk::Texture& nrmDepth = m_nrmDepth.getOutput();
  const nvvk::Texture& objCont  = m_objContour.getOutput();
  m_compositing.setInputs({color, nrmDepth, objCont});

  // Display the image of compositing
  updateDescriptor(m_compositing.getOutput().descriptor);
}


//--------------------------------------------------------------------------------------------------
// Convert all images to textures
//
void VkToonExample::importImages(tinygltf::Model& gltfModel)
{
  if(gltfModel.images.empty())
  {
    // Make dummy image(1,1), needed as we cannot have an empty array
    nvvk::ScopeCommandBuffer cmdBuf(m_device, m_graphicsQueueIndex);
    std::array<uint8_t, 4>   white = {255, 255, 255, 255};
    m_textures.emplace_back(m_alloc.createTexture(cmdBuf, 4, white.data(), nvvk::makeImage2DCreateInfo(vk::Extent2D{1, 1}), {}));
    m_debug.setObjectName(m_textures[0].image, "dummy");
    return;
  }

  m_textures.resize(gltfModel.images.size());

  vk::SamplerCreateInfo samplerCreateInfo{{}, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear};
  vk::Format            format{vk::Format::eR8G8B8A8Unorm};
  samplerCreateInfo.maxLod = FLT_MAX;

  nvvk::CommandPool sc(m_device, m_graphicsQueueIndex);

  vk::CommandBuffer cmdBuf = sc.createCommandBuffer();
  for(size_t i = 0; i < gltfModel.images.size(); i++)
  {
    auto&               gltfimage       = gltfModel.images[i];
    void*               buffer          = &gltfimage.image[0];
    VkDeviceSize        bufferSize      = gltfimage.image.size();
    auto                imgSize         = vk::Extent2D(gltfimage.width, gltfimage.height);
    vk::ImageCreateInfo imageCreateInfo = nvvk::makeImage2DCreateInfo(imgSize, format, vkIU::eSampled, true);

    nvvk::Image image = m_alloc.createImage(cmdBuf, bufferSize, buffer, imageCreateInfo);
    nvvk::cmdGenerateMipmaps(cmdBuf, image.image, format, imgSize, imageCreateInfo.mipLevels);
    vk::ImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, imageCreateInfo);
    m_textures[i]                  = m_alloc.createTexture(image, ivInfo, samplerCreateInfo);

    m_debug.setObjectName(m_textures[i].image, std::string("Txt" + std::to_string(i)).c_str());
  }
  sc.submitAndWait(cmdBuf);
  m_alloc.finalizeAndReleaseStaging();
}

//--------------------------------------------------------------------------------------------------
// Overload keyboard hit
// - Home key: fit all, the camera will move to see the entire scene bounding box
// - Space: Trigger ray picking and set the interest point at the intersection
//          also return all information under the cursor
//
void VkToonExample::onKeyboard(int key, int scancode, int action, int mods)
{
  nvvk::AppBase::onKeyboard(key, scancode, action, mods);
  if(action == GLFW_RELEASE)
    return;

  if(key == GLFW_KEY_HOME)
  {
    // Set the camera as to see the model
    if(m_gltfScene.m_cameras.empty())
      fitCamera(m_gltfScene.m_dimensions.min, m_gltfScene.m_dimensions.max, false);
    else
    {
      nvmath::vec3f eye;
      m_gltfScene.m_cameras[0].worldMatrix.get_translation(eye);
      float len = nvmath::length(m_gltfScene.m_dimensions.center - eye);
      CameraManip.setMatrix(m_gltfScene.m_cameras[0].worldMatrix, false, len);
    }
  }

  if(key == GLFW_KEY_SPACE && m_raytracer.isValid())
  {
    double x, y;
    glfwGetCursorPos(m_window, &x, &y);

    // Set the camera as to see the model
    nvvk::CommandPool sc(m_device, m_graphicsQueueIndex);
    vk::CommandBuffer cmdBuf = sc.createCommandBuffer();
    float             px     = x / float(m_size.width);
    float             py     = y / float(m_size.height);

    m_rayPicker.run(cmdBuf, px, py);
    sc.submitAndWait(cmdBuf);

    RayPicker::PickResult pr = m_rayPicker.getResult();

    if(pr.intanceID == ~0)
    {
      LOGI("Not Hit\n");
      return;
    }

    std::stringstream o;
    LOGI("\n Node:  %d", pr.intanceID);
    LOGI("\n PrimMesh:  %d", pr.intanceCustomID);
    LOGI("\n Triangle: %d", pr.primitiveID);
    LOGI("\n Distance:  %f", nvmath::length(pr.worldPos - m_sceneUbo.cameraPosition));
    LOGI("\n Position: %f, %f, %f \n", pr.worldPos.x, pr.worldPos.y, pr.worldPos.z);

    // Set the interest position
    nvmath::vec3f eye, center, up;
    CameraManip.getLookat(eye, center, up);
    CameraManip.setLookat(eye, pr.worldPos, up, false);
  }
}

//--------------------------------------------------------------------------------------------------
// Drag and dropping a glTF file
//
void VkToonExample::onFileDrop(const char* filename)
{
  namespace fs = std::filesystem;
  if(fs::path(filename).extension() != ".gltf")
  {
    LOGE("Error: only supporting .gltf files");
    return;
  }

  m_device.waitIdle();

  // Destroy all allocation: buffers, images
  m_gltfScene.destroy();
  m_alloc.destroy(m_vertexBuffer);
  m_alloc.destroy(m_normalBuffer);
  m_alloc.destroy(m_uvBuffer);
  m_alloc.destroy(m_indexBuffer);
  m_alloc.destroy(m_materialBuffer);
  m_alloc.destroy(m_sceneBuffer);
  m_alloc.destroy(m_matrixBuffer);
  for(auto& t : m_textures)
    m_alloc.destroy(t);
  m_textures.clear();

  // Destroy descriptor layout, number of images might change
  m_device.destroy(m_descSetLayout[eScene]);
  m_device.destroy(m_descPool[eScene]);

  // Destroy Raytracer data: blas, tlas, descriptorsets
  m_rayPicker.destroy();
  m_raytracer.destroy();
  m_rasterizer.destroy();

  loadScene(filename);
  settingPostPipeline();
  resetFrame();
}


//--------------------------------------------------------------------------------------------------
// IMGUI UI display
//

static void ImGuiDisabled(bool disabled)
{

  ImGui::PushItemFlag(ImGuiItemFlags_Disabled, disabled);
  if(disabled)
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
}
static void ImGuiPopDisabled(bool disabled)
{
  ImGui::PopItemFlag();
  if(disabled)
    ImGui::PopStyleVar();
}

//--------------------------------------------------------------------------------------------------
// UI rendering - Dear ImGUI
//
void VkToonExample::drawUI()
{
  bool changed = false;

  // Update imgui configuration
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
  ImGui::SetNextWindowPos(ImVec2(0, 0));

  ImGuiH::Panel::Begin();
  //  ImGui::Begin("Settings", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

  int renderer = m_useRaytracer && m_raytracer.isValid() ? 1 : 0;
  ImGuiDisabled(!m_raytracer.isValid());
  changed |= ImGui::RadioButton("Raytracer", &renderer, 1);
  ImGuiPopDisabled(!m_raytracer.isValid());
  ImGui::SameLine();
  changed |= ImGui::RadioButton("Rasterizer", &renderer, 0);
  m_useRaytracer = (renderer == 1);

  ImGui::SetNextItemOpen(true, ImGuiCond_FirstUseEver);
  if(ImGui::CollapsingHeader("Toon Setting"))
  {
    bool                        useKuwahara      = m_kuwahara.isActive();
    bool                        useAnisotropic   = m_kuwaharaAniso.isActive();
    bool                        useDepthNormal   = m_nrmDepth.isActive();
    bool                        useObjectContour = m_objContour.isActive();
    static std::array<float, 2> thetaPhiDeg      = {-45, 45};

    //--------------------------------------
    ImGui::Text("Lighting");
    changed |= ImGui::SliderInt("Steps", &m_toonNbStep, 0, 10);
    m_rasterizer.setToonSteps(m_toonNbStep);
    m_raytracer.setToonSteps(m_toonNbStep);
    changed |= ImGui::DragFloat2("Light Direction", thetaPhiDeg.data(), 1.f);
    std::array<float, 2> thetaPhiRad = {deg2rad(thetaPhiDeg[0]), deg2rad(thetaPhiDeg[1])};
    m_toonLightDir                   = nvmath::normalize(nvmath::vec3f(sin(thetaPhiRad[0]) * cos(thetaPhiRad[1]),
                                                     sin(thetaPhiRad[0]) * sin(thetaPhiRad[1]), cos(thetaPhiRad[0])));
    m_rasterizer.setToonLightDir(m_toonLightDir);
    m_raytracer.setToonLightDir(m_toonLightDir);
    //--------------------------------------
    ImGui::Separator();
    changed |= ImGui::Checkbox("Lines on Object Contour", &useObjectContour);
    ImGuiDisabled(!useObjectContour);
    changed |= m_objContour.uiSetup();
    ImGuiPopDisabled(!useObjectContour);
    //--------------------------------------
    ImGui::Separator();
    changed |= ImGui::Checkbox("Lines on Details", &useDepthNormal);
    ImGuiDisabled(!useDepthNormal);
    changed |= m_nrmDepth.uiSetup();
    ImGuiPopDisabled(!useDepthNormal);
    //--------------------------------------
    ImGui::Separator();
    changed |= ImGui::Checkbox("Image Smoothing (Kuwahara)", &useKuwahara);
    ImGuiDisabled(!useKuwahara);
    changed |= ImGui::Checkbox("Anisotropic Version", &useAnisotropic);
    if(useAnisotropic)
      changed |= m_kuwaharaAniso.uiSetup();
    else
      changed |= m_kuwahara.uiSetup();
    ImGuiPopDisabled(!useKuwahara);
    //--------------------------------------
    ImGui::Separator();
    changed |= m_compositing.uiSetup();

    m_objContour.setActive(useObjectContour);
    m_nrmDepth.setActive(useDepthNormal);
    m_kuwahara.setActive(useKuwahara);
    m_kuwaharaAniso.setActive(useAnisotropic);
  }

  changed |= ImGui::ColorEdit3("Background Color", &m_backgroundColor.x);
  m_raytracer.setClearColor(m_backgroundColor);
  m_rasterizer.setClearColor(m_backgroundColor);

  if(ImGui::CollapsingHeader("Camera"))
  {
    nvmath::vec3f eye, center, up;
    CameraManip.getLookat(eye, center, up);
    changed |= ImGui::DragFloat3("Position", &eye.x);
    changed |= ImGui::DragFloat3("Center", &center.x);
    changed |= ImGui::DragFloat3("Up", &up.x, .1f, 0.0f, 1.0f);
    float fov(CameraManip.getFov());
    if(ImGui::SliderFloat("FOV", &fov, 1, 150))
      CameraManip.setFov(fov);
    if(changed)
      CameraManip.setLookat(eye, center, up);
  }

  if(m_raytracer.isValid())
    changed |= m_raytracer.uiSetup();

  if(ImGui::CollapsingHeader("Tonemapper"))
  {
    m_tonemapper.uiSetup();
  }


  AppBase::uiDisplayHelp();

  if(changed)
  {
    settingPostPipeline();
    resetFrame();
  }

  ImGui::End();
  ImGui::Render();
}
