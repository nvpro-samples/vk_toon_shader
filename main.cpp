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


//--------------------------------------------------------------------------------------------------
// This example is creating a scene with many similar objects and a plane. There are a few materials
// and a light direction.
// More details in simple.cpp
//

#include <array>
#include <chrono>
#include <iostream>
#include <vulkan/vulkan.hpp>
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

#include "backends/imgui_impl_glfw.h"
#include "example.hpp"
#include "nvh/fileoperations.hpp"
#include "nvh/inputparser.h"
#include "nvpsystem.hpp"
#include "nvvk/context_vk.hpp"
#include "nvvk/extensions_vk.hpp"

int const SAMPLE_SIZE_WIDTH  = 1600;
int const SAMPLE_SIZE_HEIGHT = 1000;

// Default search path for shaders
std::vector<std::string> defaultSearchPaths;


//--------------------------------------------------------------------------------------------------
//
//
int main(int argc, char** argv)
{
  // setup some basic things for the sample, logging file for example
  NVPSystem system(PROJECT_NAME);

  // Default search path for shaders
  defaultSearchPaths = {
      NVPSystem::exePath() + PROJECT_NAME,
      NVPSystem::exePath() + R"(media)",
      NVPSystem::exePath() + PROJECT_RELDIRECTORY,
      NVPSystem::exePath() + PROJECT_DOWNLOAD_RELDIRECTORY,
  };

  // Parsing the command line: mandatory '-f' for the filename of the scene
  InputParser parser(argc, argv);
  std::string filename = parser.getString("-f");
  if(parser.exist("-f"))
  {
    filename = parser.getString("-f");
  }
  else if(argc == 2 && nvh::endsWith(argv[1], ".gltf"))  // Drag&Drop on .exe
  {
    filename = argv[1];
  }
  else
  {
    filename = nvh::findFile("robot/robot.gltf", defaultSearchPaths, true);
  }

  // Setup GLFW window
  if(!glfwInit())
  {
    return 1;
  }
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  GLFWwindow* window = glfwCreateWindow(SAMPLE_SIZE_WIDTH, SAMPLE_SIZE_HEIGHT, PROJECT_NAME, nullptr, nullptr);

  // Enabling the extension
  vk::PhysicalDeviceDescriptorIndexingFeaturesEXT feature;

  nvvk::ContextCreateInfo contextInfo;
  contextInfo.setVersion(1, 2);
  contextInfo.addInstanceLayer("VK_LAYER_LUNARG_monitor", true);
  contextInfo.addInstanceExtension(VK_KHR_SURFACE_EXTENSION_NAME);
#ifdef WIN32
  contextInfo.addInstanceExtension(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#else
  contextInfo.addInstanceExtension(VK_KHR_XLIB_SURFACE_EXTENSION_NAME);
  contextInfo.addInstanceExtension(VK_KHR_XCB_SURFACE_EXTENSION_NAME);
#endif
  contextInfo.addInstanceExtension(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

  contextInfo.addDeviceExtension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  contextInfo.addDeviceExtension(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME, false, &feature);
  contextInfo.addDeviceExtension(VK_KHR_MAINTENANCE3_EXTENSION_NAME, true /*optional*/);
  contextInfo.addDeviceExtension(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME, true /*optional*/);
  contextInfo.addDeviceExtension(VK_KHR_BIND_MEMORY_2_EXTENSION_NAME, true /*optional*/);
  contextInfo.addDeviceExtension(VK_NV_RAY_TRACING_EXTENSION_NAME, true /*optional*/);


  // Creating the Vulkan instance and device
  nvvk::Context vkctx;
  vkctx.initInstance(contextInfo);

  // Find all compatible devices
  auto compatibleDevices = vkctx.getCompatibleDevices(contextInfo);
  assert(!compatibleDevices.empty());

  // Use first compatible device
  vkctx.initDevice(compatibleDevices[0], contextInfo);

  VkToonExample example;

  // Window need to be opened to get the surface on which to draw
  vk::SurfaceKHR surface = example.getVkSurface(vkctx.m_instance, window);
  vkctx.setGCTQueueWithPresent(surface);

  example.setup(vkctx.m_instance, vkctx.m_device, vkctx.m_physicalDevice, vkctx.m_queueGCT.familyIndex);
  LOGI("Using %s \n", example.getPhysicalDevice().getProperties().deviceName);

  example.createSwapchain(surface, SAMPLE_SIZE_WIDTH, SAMPLE_SIZE_HEIGHT);
  example.createDepthBuffer();
  example.createRenderPass();
  example.createFrameBuffers();
  //example.createTonemapper();
  example.createAxis();
  example.createDescriptorFinal();
  example.createFinalPipeline();  // How the quad will be rendered

  example.loadScene(filename);  // Now build the example
  example.createPostProcess();  // Adding all post-effects, line extractions, ..
  example.initGUI(0);           // Using sub-pass 0


  // GLFW Callback
  example.setupGlfwCallbacks(window);
  ImGui_ImplGlfw_InitForVulkan(window, true);

  // Main loop
  while(!glfwWindowShouldClose(window))
  {
    glfwPollEvents();
    if(example.isMinimized())
      continue;

    CameraManip.updateAnim();
    example.display();  // infinitely drawing
  }
  example.destroy();
  vkctx.deinit();

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
