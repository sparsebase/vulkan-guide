
#include "vk_engine.h"

#include <SDL.h>
#include <SDL_vulkan.h>

#include <vk_types.h>
#include <vk_initializers.h>
#include <vk_descriptors.h>

#include "VkBootstrap.h"

#include <iostream>
#include <fstream>
#include <chrono>
#include <sstream>
#include "vk_textures.h"
#include "vk_shaders.h"

#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

#include "imgui.h"
#include "imgui_impl_sdl.h"
#include "imgui_impl_vulkan.h"
#include "prefab_asset.h"
#include "material_asset.h"

#include "Tracy.hpp"
#include "TracyVulkan.hpp"
#include "vk_profiler.h"

#include "fmt/core.h"
#include "fmt/os.h"
#include "fmt/color.h"

#include "logger.h"
#include "cvars.h"



AutoCVar_Int CVAR_OcclusionCullGPU("culling.enableOcclusionGPU", "Perform occlusion culling in gpu", 1, CVarFlags::EditCheckbox);


AutoCVar_Int CVAR_CamLock("camera.lock", "Locks the camera", 0, CVarFlags::EditCheckbox);
AutoCVar_Int CVAR_OutputIndirectToFile("culling.outputIndirectBufferToFile", "output the indirect data to a file. Autoresets", 0, CVarFlags::EditCheckbox);

AutoCVar_Float CVAR_DrawDistance("gpu.drawDistance", "Distance cull", 5000);

AutoCVar_Int CVAR_FreezeShadows("gpu.freezeShadows", "Stop the rendering of shadows", 0, CVarFlags::EditCheckbox);


constexpr bool bUseValidationLayers = false;

//we want to immediately abort when there is an error. In normal engines this would give an error message to the user, or perform a dump of state.
using namespace std;
#define VK_CHECK(x)                                                 \
	do                                                              \
	{                                                               \
		VkResult err = x;                                           \
		if (err)                                                    \
		{                                                           \
			std::cout <<"Detected Vulkan error: " << err << std::endl; \
			abort();                                                \
		}                                                           \
	} while (0)




void VulkanEngine::init()
{
	ZoneScopedN("Engine Init");	
	
	LogHandler::Get().set_time();	

	LOG_INFO("Engine Init");

	// We initialize SDL and create a window with it. 
	SDL_Init(SDL_INIT_VIDEO);
	LOG_SUCCESS("SDL inited");
	SDL_WindowFlags window_flags = (SDL_WINDOW_VULKAN);

	window_ = SDL_CreateWindow(
		"",
		SDL_WINDOWPOS_UNDEFINED,
		SDL_WINDOWPOS_UNDEFINED,
		windowExtent_.width,
		windowExtent_.height,
		window_flags
	);

	//_renderables.reserve(10000);
	
	meshes_.reserve(1000);
	
	init_vulkan();

	profiler_ = new vkutil::VulkanProfiler();

	profiler_->init(device_, gpuProperties_.limits.timestampPeriod);

	shaderCache_.init(device_);

	renderScene_.init();

	init_swapchain();


	init_forward_renderpass();
	init_copy_renderpass();
	init_shadow_renderpass();

	init_framebuffers();

	init_commands();

	init_sync_structures();

	init_descriptors();

	init_pipelines();

	LOG_INFO("Engine Initialized, starting Load");
	

	load_meshes();

	load_images();

	init_scene();

	init_imgui();
	
	renderScene_.build_batches();

	renderScene_.merge_meshes(this);
	//everything went fine
	isInitialized_ = true;

	camera_ = {};
	camera_.position = { 0.f,6.f,5.f };

	mainLight_.lightPosition = { 0,0,0 };
	mainLight_.lightDirection = glm::vec3(0.3, -1, 0.3);
	mainLight_.shadowExtent = { 100 ,100 ,100 };
}
void VulkanEngine::cleanup()
{
	if (isInitialized_) {

		//make sure the gpu has stopped doing its things
		for (auto& frame : frames_)
		{
			vkWaitForFences(device_, 1, &frame.renderFence_, true, 1000000000);
		}

		mainDeletionQueue_.flush();

		for (auto& frame : frames_)
		{
			frame.dynamicDescriptorAllocator_->cleanup();
		}

		descriptorAllocator_->cleanup();
		descriptorLayoutCache_->cleanup();


		vkDestroySurfaceKHR(instance_, surface_, nullptr);

		vkDestroyDevice(device_, nullptr);
		vkDestroyInstance(instance_, nullptr);

		SDL_DestroyWindow(window_);
	}
}

void VulkanEngine::draw()
{
	ZoneScopedN("Engine Draw");

	ImGui::Render();

	{
		ZoneScopedN("Fence Wait");
		//wait until the gpu has finished rendering the last frame. Timeout of 1 second
		VK_CHECK(vkWaitForFences(device_, 1, &get_current_frame().renderFence_, true, 1000000000));
		VK_CHECK(vkResetFences(device_, 1, &get_current_frame().renderFence_));

		get_current_frame().dynamicData_.reset();

		renderScene_.build_batches();
		//check the debug data
		void* data;		
		vmaMapMemory(allocator_, get_current_frame().debugOutputBuffer_.allocation_, &data);
		for (int i =1 ; i <   get_current_frame().debugDataNames_.size();i++)
		{
			uint32_t begin = get_current_frame().debugDataOffsets_[i-1];
			uint32_t end = get_current_frame().debugDataOffsets_[i];

			auto name = get_current_frame().debugDataNames_[i];
			if (name.compare("Cull Indirect Output") == 0)
			{
				void* buffer = malloc(end - begin);
				memcpy(buffer, (uint8_t*)data + begin, end - begin);

				GPUIndirectObject* objects = (GPUIndirectObject*)buffer;
				int objectCount = (end - begin) / sizeof(GPUIndirectObject);

				std::string filename = fmt::format("{}_CULLDATA_{}.txt", frameNumber_,i);

				auto out = fmt::output_file(filename);

				for (int o = 0; o < objectCount; o++)
				{
					out.print("DRAW: {} ------------ \n", o);
					out.print("	OG Count: {} \n", renderScene_.forwardPass_.batches_[o].count);
					out.print("	Visible Count: {} \n", objects[o].command.instanceCount);
					out.print("	First: {} \n", objects[o].command.firstInstance);
					out.print("	Indices: {} \n", objects[o].command.indexCount);
				}
				
				free(buffer);
			}
		}

		vmaUnmapMemory(allocator_, get_current_frame().debugOutputBuffer_.allocation_);
		get_current_frame().debugDataNames_.clear();
		get_current_frame().debugDataOffsets_.clear();

		get_current_frame().debugDataNames_.push_back("");
		get_current_frame().debugDataOffsets_.push_back(0);
	}
	get_current_frame().frameDeletionQueue_.flush();
	get_current_frame().dynamicDescriptorAllocator_->reset_pools();

	//now that we are sure that the commands finished executing, we can safely reset the command buffer to begin recording again.
	VK_CHECK(vkResetCommandBuffer(get_current_frame().mainCommandBuffer_, 0));
	uint32_t swapchainImageIndex;
	{
		ZoneScopedN("Aquire Image");
		//request image from the swapchain

		VK_CHECK(vkAcquireNextImageKHR(device_, swapchain_, 0, get_current_frame().presentSemaphore_, nullptr, &swapchainImageIndex));

	}

	//naming it cmd for shorter writing
	VkCommandBuffer cmd = get_current_frame().mainCommandBuffer_;

	//begin the command buffer recording. We will use this command buffer exactly once, so we want to let vulkan know that
	VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

	VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

	//make a clear-color from frame number. This will flash with a 120 frame period.
	VkClearValue clearValue;
	float flash = abs(sin(frameNumber_ / 120.f));
	clearValue.color = { { 0.1f, 0.1f, 0.1f, 1.0f } };

	profiler_->grab_queries(cmd);

	{

		postCullBarriers_.clear();
		cullReadyBarriers_.clear();

		TracyVkZone(graphicsQueueContext_, get_current_frame().mainCommandBuffer_, "All Frame");
		ZoneScopedNC("Render Frame", tracy::Color::White);

		vkutil::VulkanScopeTimer timer(cmd, profiler_, "All Frame");

		{
			vkutil::VulkanScopeTimer timer2(cmd, profiler_, "Ready Frame");

			ready_mesh_draw(cmd);

			ready_cull_data(renderScene_.forwardPass_, cmd);
			ready_cull_data(renderScene_.transparentForwardPass_, cmd);
			ready_cull_data(renderScene_.shadowPass_, cmd);

			vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, cullReadyBarriers_.size(), cullReadyBarriers_.data(), 0, nullptr);
		}


		CullParams forwardCull;
		forwardCull.projmat = camera_.get_projection_matrix(true);
		forwardCull.viewmat = camera_.get_view_matrix();
		forwardCull.frustrumCull = true;
		forwardCull.occlusionCull = true;
		forwardCull.drawDist = CVAR_DrawDistance.Get();
		forwardCull.aabb = false;
		{
			execute_compute_cull(cmd, renderScene_.forwardPass_, forwardCull);
			execute_compute_cull(cmd, renderScene_.transparentForwardPass_, forwardCull);
		}

		glm::vec3 extent = mainLight_.shadowExtent * 10.f;
		glm::mat4 projection = glm::orthoLH_ZO(-extent.x, extent.x, -extent.y, extent.y, -extent.z, extent.z);
		
		
		CullParams shadowCull;
		shadowCull.projmat = mainLight_.get_projection();
		shadowCull.viewmat = mainLight_.get_view();
		shadowCull.frustrumCull = true;
		shadowCull.occlusionCull = false;
		shadowCull.drawDist = 9999999;
		shadowCull.aabb = true;

		glm::vec3 aabbcenter = mainLight_.lightPosition;
		glm::vec3 aabbextent = mainLight_.shadowExtent * 1.5f;
		shadowCull.aabbmax = aabbcenter + aabbextent;
		shadowCull.aabbmin = aabbcenter - aabbextent;

		{
			vkutil::VulkanScopeTimer timer2(cmd, profiler_, "Shadow Cull");

			if (*CVarSystem::Get()->GetIntCVar("gpu.shadowcast"))
			{
				execute_compute_cull(cmd, renderScene_.shadowPass_, shadowCull);
			}
		}

		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 0, nullptr, postCullBarriers_.size(), postCullBarriers_.data(), 0, nullptr);



		shadow_pass(cmd);
		
		forward_pass(clearValue, cmd);

		reduce_depth(cmd);

		copy_render_to_swapchain(swapchainImageIndex, cmd);
	}

	TracyVkCollect(graphicsQueueContext_, get_current_frame().mainCommandBuffer_);

	//finalize the command buffer (we can no longer add commands, but it can now be executed)
	VK_CHECK(vkEndCommandBuffer(cmd));

	//prepare the submission to the queue. 
	//we want to wait on the _presentSemaphore, as that semaphore is signaled when the swapchain is ready
	//we will signal the _renderSemaphore, to signal that rendering has finished

	VkSubmitInfo submit = vkinit::submit_info(&cmd);
	VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

	submit.pWaitDstStageMask = &waitStage;

	submit.waitSemaphoreCount = 1;
	submit.pWaitSemaphores = &get_current_frame().presentSemaphore_;

	submit.signalSemaphoreCount = 1;
	submit.pSignalSemaphores = &get_current_frame().renderSemaphore_;
	{
		ZoneScopedN("Queue Submit");
		//submit command buffer to the queue and execute it.
		// _renderFence will now block until the graphic commands finish execution
		VK_CHECK(vkQueueSubmit(graphicsQueue_, 1, &submit, get_current_frame().renderFence_));

	}
	//prepare present
	// this will put the image we just rendered to into the visible window.
	// we want to wait on the _renderSemaphore for that, 
	// as its necessary that drawing commands have finished before the image is displayed to the user
	VkPresentInfoKHR presentInfo = vkinit::present_info();

	presentInfo.pSwapchains = &swapchain_;
	presentInfo.swapchainCount = 1;

	presentInfo.pWaitSemaphores = &get_current_frame().renderSemaphore_;
	presentInfo.waitSemaphoreCount = 1;

	presentInfo.pImageIndices = &swapchainImageIndex;

	{
		ZoneScopedN("Queue Present");
		VK_CHECK(vkQueuePresentKHR(graphicsQueue_, &presentInfo));

	}
	//increase the number of frames drawn
	frameNumber_++;
}


void VulkanEngine::forward_pass(VkClearValue clearValue, VkCommandBuffer cmd)
{
	vkutil::VulkanScopeTimer timer(cmd, profiler_, "Forward Pass");
	vkutil::VulkanPipelineStatRecorder timer2(cmd, profiler_, "Forward Primitives");
	//clear depth at 0
	VkClearValue depthClear;
	depthClear.depthStencil.depth = 0.f;

	//start the main renderpass. 
	//We will use the clear color from above, and the framebuffer of the index the swapchain gave us
	VkRenderPassBeginInfo rpInfo = vkinit::renderpass_begin_info(renderPass_, windowExtent_, forwardFramebuffer_/*_framebuffers[swapchainImageIndex]*/);

	//connect clear values
	rpInfo.clearValueCount = 2;

	VkClearValue clearValues[] = { clearValue, depthClear };

	rpInfo.pClearValues = &clearValues[0];
	vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

	VkViewport viewport;
	viewport.x = 0.0f;
	viewport.y = 0.0f;
	viewport.width = (float)windowExtent_.width;
	viewport.height = (float)windowExtent_.height;
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;

	VkRect2D scissor;
	scissor.offset = { 0, 0 };
	scissor.extent = windowExtent_;

	vkCmdSetViewport(cmd, 0, 1, &viewport);
	vkCmdSetScissor(cmd, 0, 1, &scissor);
	vkCmdSetDepthBias(cmd, 0, 0, 0);


	//stats.drawcalls = 0;
	//stats.draws = 0;
	//stats.objects = 0;
	//stats.triangles = 0;

	{
		TracyVkZone(graphicsQueueContext_, get_current_frame().mainCommandBuffer_, "Forward Pass");
		draw_objects_forward(cmd, renderScene_.forwardPass_);
		draw_objects_forward(cmd, renderScene_.transparentForwardPass_);
	}


	{
		TracyVkZone(graphicsQueueContext_, get_current_frame().mainCommandBuffer_, "Imgui Draw");
		ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
	}

	//finalize the render pass
	vkCmdEndRenderPass(cmd);
}


void VulkanEngine::shadow_pass(VkCommandBuffer cmd)
{
	

	vkutil::VulkanScopeTimer timer(cmd, profiler_, "Shadow Pass");
	vkutil::VulkanPipelineStatRecorder timer2(cmd, profiler_, "Shadow Primitives");
	if (CVAR_FreezeShadows.Get()) return;
	if (!*CVarSystem::Get()->GetIntCVar("gpu.shadowcast"))
	{
		return;
	}

	//clear depth at 1
	VkClearValue depthClear;
	depthClear.depthStencil.depth = 1.f;	
	VkRenderPassBeginInfo rpInfo = vkinit::renderpass_begin_info(shadowPass_, shadowExtent_, shadowFramebuffer_);

	//connect clear values
	rpInfo.clearValueCount = 1;

	VkClearValue clearValues[] = { depthClear };

	rpInfo.pClearValues = &clearValues[0];
	vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

	VkViewport viewport;
	viewport.x = 0.0f;
	viewport.y = 0.0f;
	viewport.width = (float)shadowExtent_.width;
	viewport.height = (float)shadowExtent_.height;
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;

	VkRect2D scissor;
	scissor.offset = { 0, 0 };
	scissor.extent = shadowExtent_;

	vkCmdSetViewport(cmd, 0, 1, &viewport);
	vkCmdSetScissor(cmd, 0, 1, &scissor);
	


	stats.drawcalls = 0;
	stats.draws = 0;
	stats.objects = 0;
	stats.triangles = 0;

	if(renderScene_.shadowPass_.batches_.size() > 0)
	{
		TracyVkZone(graphicsQueueContext_, get_current_frame().mainCommandBuffer_, "Shadow  Pass");
		draw_objects_shadow(cmd, renderScene_.shadowPass_);
	}

	//finalize the render pass
	vkCmdEndRenderPass(cmd);
}

void VulkanEngine::copy_render_to_swapchain(uint32_t swapchainImageIndex, VkCommandBuffer cmd)
{
	//start the main renderpass. 
	//We will use the clear color from above, and the framebuffer of the index the swapchain gave us
	VkRenderPassBeginInfo copyRP = vkinit::renderpass_begin_info(copyPass_, windowExtent_, framebuffers_[swapchainImageIndex]);


	vkCmdBeginRenderPass(cmd, &copyRP, VK_SUBPASS_CONTENTS_INLINE);

	VkViewport viewport;
	viewport.x = 0.0f;
	viewport.y = 0.0f;
	viewport.width = (float)windowExtent_.width;
	viewport.height = (float)windowExtent_.height;
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;

	VkRect2D scissor;
	scissor.offset = { 0, 0 };
	scissor.extent = windowExtent_;

	vkCmdSetViewport(cmd, 0, 1, &viewport);
	vkCmdSetScissor(cmd, 0, 1, &scissor);

	vkCmdSetDepthBias(cmd, 0, 0, 0);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, blitPipeline_);

	VkDescriptorImageInfo sourceImage;
	sourceImage.sampler = smoothSampler_;

	sourceImage.imageView = rawRenderImage_.defaultView_;
	sourceImage.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

	VkDescriptorSet blitSet;
	vkutil::DescriptorBuilder::begin(descriptorLayoutCache_, get_current_frame().dynamicDescriptorAllocator_)
		.bind_image(0, &sourceImage, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
		.build(blitSet);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, blitLayout_, 0, 1, &blitSet, 0, nullptr);

	vkCmdDraw(cmd, 3, 1, 0, 0);


	vkCmdEndRenderPass(cmd);
}

void VulkanEngine::run()
{

	LOG_INFO("Starting Main Loop ");
	
	bool bQuit = false;

	// Using time point and system_clock 
	std::chrono::time_point<std::chrono::system_clock> start, end;
	
	start = std::chrono::system_clock::now();
	end = std::chrono::system_clock::now();
	//main loop
	while (!bQuit)
	{
		ZoneScopedN("Main Loop");
		end = std::chrono::system_clock::now();
		std::chrono::duration<float> elapsed_seconds = end - start;
		stats.frametime = elapsed_seconds.count() * 1000.f;

		start = std::chrono::system_clock::now();
		//Handle events on queue
		SDL_Event e;
		{
			ZoneScopedNC("Event Loop", tracy::Color::White);
			while (SDL_PollEvent(&e) != 0)
			{

				ImGui_ImplSDL2_ProcessEvent(&e);
				camera_.process_input_event(&e);


				//close the window when user alt-f4s or clicks the X button			
				if (e.type == SDL_QUIT)
				{
					bQuit = true;
				}
				else if (e.type == SDL_KEYDOWN)
				{
					if (e.key.keysym.sym == SDLK_SPACE)
					{
						selectedShader_ += 1;
						if (selectedShader_ > 1)
						{
							selectedShader_ = 0;
						}
					}
					if (e.key.keysym.sym == SDLK_TAB)
					{
						if (CVAR_CamLock.Get())
						{
							LOG_INFO("Mouselook disabled");
							CVAR_CamLock.Set(false);
						}
						else {
							LOG_INFO("Mouselook enabled");
							CVAR_CamLock.Set(true);
						}
					}
				}
			}
		}
		{
			ZoneScopedNC("Imgui Logic", tracy::Color::Grey);

			//imgui new frame 
			ImGui_ImplVulkan_NewFrame();
			ImGui_ImplSDL2_NewFrame(window_);

			ImGui::NewFrame();

			if (ImGui::BeginMainMenuBar())
			{
				if (ImGui::BeginMenu("Debug"))
				{
					if (ImGui::BeginMenu("CVAR"))
					{
						CVarSystem::Get()->DrawImguiEditor();
						ImGui::EndMenu();
					}
					ImGui::EndMenu();
				}
				ImGui::EndMainMenuBar();
			}


			ImGui::Begin("engine");

			ImGui::Text("Frametimes: %f", stats.frametime);
			ImGui::Text("Objects: %d", stats.objects);
			//ImGui::Text("Drawcalls: %d", stats.drawcalls);
			ImGui::Text("Batches: %d", stats.draws);
			//ImGui::Text("Triangles: %d", stats.triangles);		
			
			CVAR_OutputIndirectToFile.Set(false);
			if (ImGui::Button("Output Indirect"))
			{
				CVAR_OutputIndirectToFile.Set(true);
			}
		

			ImGui::Separator();

			for (auto& [k, v] : profiler_->timing)
			{
				ImGui::Text("TIME %s %f ms",k.c_str(), v);
			}
			for (auto& [k, v] : profiler_->stats)
			{
				ImGui::Text("STAT %s %d", k.c_str(), v);
			}
			ImGui::Text("Camera pos: [%.2f, %.2f, %.2f]", camera_.position.x, camera_.position.y, camera_.position.z);

			ImGui::End();
		}

		{
			ZoneScopedNC("Flag Objects", tracy::Color::Blue);
			//test flagging some objects for changes

// 			int N_changes = 1000;
// 			for (int i = 0; i < N_changes; i++)
// 			{
// 				int rng = rand() % _renderScene.renderables.size();
// 
// 				Handle<RenderObject> h;
// 				h.handle = rng;
// 				_renderScene.update_object(h);
// 			}
			camera_.bLocked = CVAR_CamLock.Get();

			camera_.update_camera(stats.frametime);

			//_mainLight.lightPosition = _camera.position;
			mainLight_.lightPosition = glm::vec3(0, 1624, 0);
		}
	
		draw();
	}
}

FrameData& VulkanEngine::get_current_frame()
{
	return frames_[frameNumber_ % FRAME_OVERLAP];
}


FrameData& VulkanEngine::get_last_frame()
{
	return frames_[(frameNumber_ - 1) % 2];
}


void VulkanEngine::process_input_event(SDL_Event* ev)
{
	if (ev->type == SDL_KEYDOWN)
	{
		switch (ev->key.keysym.sym)
		{		
		
		}
	}
	else if (ev->type == SDL_KEYUP)
	{
		switch (ev->key.keysym.sym)
		{
		case SDLK_UP:
		case SDLK_w:
			camera_.inputAxis.x -= 1.f;
			break;
		case SDLK_DOWN:
		case SDLK_s:
			camera_.inputAxis.x += 1.f;
			break;
		case SDLK_LEFT:
		case SDLK_a:
			camera_.inputAxis.y += 1.f;
			break;
		case SDLK_RIGHT:
		case SDLK_d:
			camera_.inputAxis.y -= 1.f;
			break;
		}
	}
	else if (ev->type == SDL_MOUSEMOTION) {
		if (!CVAR_CamLock.Get())
		{
			camera_.pitch -= ev->motion.yrel * 0.003f;
			camera_.yaw -= ev->motion.xrel * 0.003f;
		}
	}

	camera_.inputAxis = glm::clamp(camera_.inputAxis, { -1.0,-1.0,-1.0 }, { 1.0,1.0,1.0 });
}

void VulkanEngine::init_vulkan()
{
	
	vkb::InstanceBuilder builder;
	//make the vulkan instance, with basic debug features
	auto inst_ret = builder.set_app_name("Example Vulkan Application")

		.request_validation_layers(bUseValidationLayers)
		.use_default_debug_messenger()
		.build();


	LOG_SUCCESS("Vulkan Instance initialized");

	vkb::Instance vkb_inst = inst_ret.value();

	//grab the instance 
	instance_ = vkb_inst.instance;

	SDL_Vulkan_CreateSurface(window_, instance_, &surface_);

	LOG_SUCCESS("SDL Surface initialized");

	//use vkbootstrap to select a gpu. 
	//We want a gpu that can write to the SDL surface and supports vulkan 1.2
	vkb::PhysicalDeviceSelector selector{ vkb_inst };
	VkPhysicalDeviceFeatures feats{};

	feats.pipelineStatisticsQuery = true;
	feats.multiDrawIndirect = true;
	feats.drawIndirectFirstInstance = true;
	feats.samplerAnisotropy = true;
	selector.set_required_features(feats);

	vkb::PhysicalDevice physicalDevice = selector
		.set_minimum_version(1, 1)
		.set_surface(surface_)
		.add_required_extension(VK_EXT_SAMPLER_FILTER_MINMAX_EXTENSION_NAME)
		
		.select()
		.value();

	LOG_SUCCESS("GPU found");

	//create the final vulkan device

	vkb::DeviceBuilder deviceBuilder{ physicalDevice };


	

	vkb::Device vkbDevice = deviceBuilder.build().value();
	
	// Get the VkDevice handle used in the rest of a vulkan application
	device_ = vkbDevice.device;
	physicalDevice_ = physicalDevice.physical_device;

	// use vkbootstrap to get a Graphics queue
	graphicsQueue_ = vkbDevice.get_queue(vkb::QueueType::graphics).value();

	graphicsQueueFamily_ = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

	//initialize the memory allocator
	VmaAllocatorCreateInfo allocatorInfo = {};
	allocatorInfo.physicalDevice = physicalDevice_;
	allocatorInfo.device = device_;
	allocatorInfo.instance = instance_;
	vmaCreateAllocator(&allocatorInfo, &allocator_);


	
	vkGetPhysicalDeviceProperties(physicalDevice_, &gpuProperties_);

	LOG_INFO("The gpu has a minimum buffer alignement of {}", gpuProperties_.limits.minUniformBufferOffsetAlignment);
}
uint32_t previousPow2(uint32_t v)
{
	uint32_t r = 1;

	while (r * 2 < v)
		r *= 2;

	return r;
}
uint32_t getImageMipLevels(uint32_t width, uint32_t height)
{
	uint32_t result = 1;

	while (width > 1 || height > 1)
	{
		result++;
		width /= 2;
		height /= 2;
	}

	return result;
}
void VulkanEngine::init_swapchain()
{

	vkb::SwapchainBuilder swapchainBuilder{ physicalDevice_,device_,surface_ };

	vkb::Swapchain vkbSwapchain = swapchainBuilder
		.use_default_format_selection()
		//use vsync present mode
		.set_desired_present_mode(VK_PRESENT_MODE_MAILBOX_KHR)
		.set_desired_extent(windowExtent_.width, windowExtent_.height)
		
		.build()
		.value();

	//store swapchain and its related images
	swapchain_ = vkbSwapchain.swapchain;
	swapchainImages_ = vkbSwapchain.get_images().value();
	swapchainImageViews_ = vkbSwapchain.get_image_views().value();

	swachainImageFormat_ = vkbSwapchain.image_format;

	//render image
	{
		//depth image size will match the window
		VkExtent3D renderImageExtent = {
			windowExtent_.width,
			windowExtent_.height,
			1
		};
		renderFormat_ = VK_FORMAT_R32G32B32A32_SFLOAT;
		VkImageCreateInfo ri_info = vkinit::image_create_info(renderFormat_, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT| VK_IMAGE_USAGE_SAMPLED_BIT, renderImageExtent);

		//for the depth image, we want to allocate it from gpu local memory
		VmaAllocationCreateInfo dimg_allocinfo = {};
		dimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
		dimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

		//allocate and create the image
		vmaCreateImage(allocator_, &ri_info, &dimg_allocinfo, &rawRenderImage_.image_, &rawRenderImage_.allocation_, nullptr);

		//build a image-view for the depth image to use for rendering
		VkImageViewCreateInfo dview_info = vkinit::imageview_create_info(renderFormat_, rawRenderImage_.image_, VK_IMAGE_ASPECT_COLOR_BIT);

		VK_CHECK(vkCreateImageView(device_, &dview_info, nullptr, &rawRenderImage_.defaultView_));
	}


	mainDeletionQueue_.push_function([=]() {
		vkDestroySwapchainKHR(device_, swapchain_, nullptr);
		});

	//depth image size will match the window
	VkExtent3D depthImageExtent = {
		windowExtent_.width,
		windowExtent_.height,
		1
	};

	VkExtent3D shadowExtent = {
		shadowExtent_.width,
		shadowExtent_.height,
		1
	};

	//hardcoding the depth format to 32 bit float
	depthFormat_ = VK_FORMAT_D32_SFLOAT;

	//for the depth image, we want to allocate it from gpu local memory
	VmaAllocationCreateInfo dimg_allocinfo = {};
	dimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
	dimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	// depth image ------ 
	{
		//the depth image will be a image with the format we selected and Depth Attachment usage flag
		VkImageCreateInfo dimg_info = vkinit::image_create_info(depthFormat_, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, depthImageExtent);


		//allocate and create the image
		vmaCreateImage(allocator_, &dimg_info, &dimg_allocinfo, &depthImage_.image_, &depthImage_.allocation_, nullptr);


		//build a image-view for the depth image to use for rendering
		VkImageViewCreateInfo dview_info = vkinit::imageview_create_info(depthFormat_, depthImage_.image_, VK_IMAGE_ASPECT_DEPTH_BIT);;

		VK_CHECK(vkCreateImageView(device_, &dview_info, nullptr, &depthImage_.defaultView_));
	}
	//shadow image
	{
		//the depth image will be a image with the format we selected and Depth Attachment usage flag
		VkImageCreateInfo dimg_info = vkinit::image_create_info(depthFormat_, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, shadowExtent);

		//allocate and create the image
		vmaCreateImage(allocator_, &dimg_info, &dimg_allocinfo, &shadowImage_.image_, &shadowImage_.allocation_, nullptr);

		//build a image-view for the depth image to use for rendering
		VkImageViewCreateInfo dview_info = vkinit::imageview_create_info(depthFormat_, shadowImage_.image_, VK_IMAGE_ASPECT_DEPTH_BIT);

		VK_CHECK(vkCreateImageView(device_, &dview_info, nullptr, &shadowImage_.defaultView_));
	}


	// Note: previousPow2 makes sure all reductions are at most by 2x2 which makes sure they are conservative
	depthPyramidWidth_ = previousPow2(windowExtent_.width);
	depthPyramidHeight_ = previousPow2(windowExtent_.height);
	depthPyramidLevels_ = getImageMipLevels(depthPyramidWidth_, depthPyramidHeight_);

	VkExtent3D pyramidExtent = {
		static_cast<uint32_t>(depthPyramidWidth_),
		static_cast<uint32_t>(depthPyramidHeight_),
		1
	};
	//the depth image will be a image with the format we selected and Depth Attachment usage flag
	VkImageCreateInfo pyramidInfo = vkinit::image_create_info(VK_FORMAT_R32_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, pyramidExtent);

	pyramidInfo.mipLevels = depthPyramidLevels_;

	//allocate and create the image
	vmaCreateImage(allocator_, &pyramidInfo, &dimg_allocinfo, &depthPyramid_.image_, &depthPyramid_.allocation_, nullptr);

	//build a image-view for the depth image to use for rendering
	VkImageViewCreateInfo priview_info = vkinit::imageview_create_info(VK_FORMAT_R32_SFLOAT, depthPyramid_.image_, VK_IMAGE_ASPECT_COLOR_BIT);
	priview_info.subresourceRange.levelCount = depthPyramidLevels_;


	VK_CHECK(vkCreateImageView(device_, &priview_info, nullptr, &depthPyramid_.defaultView_));


	for (int32_t i = 0; i < depthPyramidLevels_; ++i)
	{
		VkImageViewCreateInfo level_info = vkinit::imageview_create_info(VK_FORMAT_R32_SFLOAT, depthPyramid_.image_, VK_IMAGE_ASPECT_COLOR_BIT);
		level_info.subresourceRange.levelCount = 1;
		level_info.subresourceRange.baseMipLevel = i;

		VkImageView pyramid;
		vkCreateImageView(device_, &level_info, nullptr, &pyramid);

		depthPyramidMips_[i] = pyramid;
		assert(depthPyramidMips_[i]);
	}




	
	VkSamplerCreateInfo createInfo = {};

	auto reductionMode = VK_SAMPLER_REDUCTION_MODE_MIN;

	createInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	createInfo.magFilter = VK_FILTER_LINEAR;
	createInfo.minFilter = VK_FILTER_LINEAR;
	createInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
	createInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	createInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	createInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	createInfo.minLod = 0;
	createInfo.maxLod = 16.f;

	VkSamplerReductionModeCreateInfoEXT createInfoReduction = { VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO_EXT };

	if (reductionMode != VK_SAMPLER_REDUCTION_MODE_WEIGHTED_AVERAGE_EXT)
	{
		createInfoReduction.reductionMode = reductionMode;

		createInfo.pNext = &createInfoReduction;
	}

	
	VK_CHECK(vkCreateSampler(device_, &createInfo, 0, &depthSampler_));

	VkSamplerCreateInfo samplerInfo = vkinit::sampler_create_info(VK_FILTER_LINEAR);
	samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	
	vkCreateSampler(device_, &samplerInfo, nullptr, &smoothSampler_);

	VkSamplerCreateInfo shadsamplerInfo = vkinit::sampler_create_info(VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER);
	shadsamplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
	shadsamplerInfo.compareEnable = true;
	shadsamplerInfo.compareOp = VK_COMPARE_OP_LESS;
	vkCreateSampler(device_, &shadsamplerInfo, nullptr, &shadowSampler_);


	//add to deletion queues
	mainDeletionQueue_.push_function([=]() {
		vkDestroyImageView(device_, depthImage_.defaultView_, nullptr);
		vmaDestroyImage(allocator_, depthImage_.image_, depthImage_.allocation_);
		});
}

void VulkanEngine::init_forward_renderpass()
{
	//we define an attachment description for our main color image
	//the attachment is loaded as "clear" when renderpass start
	//the attachment is stored when renderpass ends
	//the attachment layout starts as "undefined", and transitions to "Present" so its possible to display it
	//we dont care about stencil, and dont use multisampling

	VkAttachmentDescription color_attachment = {};
	color_attachment.format = renderFormat_;//_swachainImageFormat;
	color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
	color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	color_attachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;//PRESENT_SRC_KHR;

	VkAttachmentReference color_attachment_ref = {};
	color_attachment_ref.attachment = 0;
	color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkAttachmentDescription depth_attachment = {};
	// Depth attachment
	depth_attachment.flags = 0;
	depth_attachment.format = depthFormat_;
	depth_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
	depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	depth_attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkAttachmentReference depth_attachment_ref = {};
	depth_attachment_ref.attachment = 1;
	depth_attachment_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	//we are going to create 1 subpass, which is the minimum you can do
	VkSubpassDescription subpass = {};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &color_attachment_ref;
	//hook the depth attachment into the subpass
	subpass.pDepthStencilAttachment = &depth_attachment_ref;

	//1 dependency, which is from "outside" into the subpass. And we can read or write color
	VkSubpassDependency dependency = {};
	dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
	dependency.dstSubpass = 0;
	dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.srcAccessMask = 0;
	dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;


	//array of 2 attachments, one for the color, and other for depth
	VkAttachmentDescription attachments[2] = { color_attachment,depth_attachment };

	VkRenderPassCreateInfo render_pass_info = {};
	render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	//2 attachments from said array
	render_pass_info.attachmentCount = 2;
	render_pass_info.pAttachments = &attachments[0];
	render_pass_info.subpassCount = 1;
	render_pass_info.pSubpasses = &subpass;
	//render_pass_info.dependencyCount = 1;
	//render_pass_info.pDependencies = &dependency;

	VK_CHECK(vkCreateRenderPass(device_, &render_pass_info, nullptr, &renderPass_));

	mainDeletionQueue_.push_function([=]() {
		vkDestroyRenderPass(device_, renderPass_, nullptr);
		});
}


void VulkanEngine::init_copy_renderpass()
{
	//we define an attachment description for our main color image
//the attachment is loaded as "clear" when renderpass start
//the attachment is stored when renderpass ends
//the attachment layout starts as "undefined", and transitions to "Present" so its possible to display it
//we dont care about stencil, and dont use multisampling

	VkAttachmentDescription color_attachment = {};
	color_attachment.format = swachainImageFormat_;
	color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
	color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

	VkAttachmentReference color_attachment_ref = {};
	color_attachment_ref.attachment = 0;
	color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	//we are going to create 1 subpass, which is the minimum you can do
	VkSubpassDescription subpass = {};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &color_attachment_ref;	

	//1 dependency, which is from "outside" into the subpass. And we can read or write color
	VkSubpassDependency dependency = {};
	dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
	dependency.dstSubpass = 0;
	dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.srcAccessMask = 0;
	dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;


	VkRenderPassCreateInfo render_pass_info = {};
	render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	//2 attachments from said array
	render_pass_info.attachmentCount = 1;
	render_pass_info.pAttachments = &color_attachment;
	render_pass_info.subpassCount = 1;
	render_pass_info.pSubpasses = &subpass;
	//render_pass_info.dependencyCount = 1;
	//render_pass_info.pDependencies = &dependency;

	VK_CHECK(vkCreateRenderPass(device_, &render_pass_info, nullptr, &copyPass_));

	mainDeletionQueue_.push_function([=]() {
		vkDestroyRenderPass(device_, copyPass_, nullptr);
		});
}


void VulkanEngine::init_shadow_renderpass()
{
	VkAttachmentDescription depth_attachment = {};
	// Depth attachment
	depth_attachment.flags = 0;
	depth_attachment.format = depthFormat_;
	depth_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
	depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	depth_attachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

	VkAttachmentReference depth_attachment_ref = {};
	depth_attachment_ref.attachment =0;
	depth_attachment_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	//we are going to create 1 subpass, which is the minimum you can do
	VkSubpassDescription subpass = {};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;

	//hook the depth attachment into the subpass
	subpass.pDepthStencilAttachment = &depth_attachment_ref;

	//1 dependency, which is from "outside" into the subpass. And we can read or write color
	VkSubpassDependency dependency = {};
	dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
	dependency.dstSubpass = 0;
	dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.srcAccessMask = 0;
	dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

	VkRenderPassCreateInfo render_pass_info = {};
	render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	//2 attachments from said array
	render_pass_info.attachmentCount = 1;
	render_pass_info.pAttachments = &depth_attachment;
	render_pass_info.subpassCount = 1;
	render_pass_info.pSubpasses = &subpass;	

	VK_CHECK(vkCreateRenderPass(device_, &render_pass_info, nullptr, &shadowPass_));

	mainDeletionQueue_.push_function([=]() {
		vkDestroyRenderPass(device_, shadowPass_, nullptr);
	});
}

void VulkanEngine::init_framebuffers()
{
	

	const uint32_t swapchain_imagecount = static_cast<uint32_t>(swapchainImages_.size());
	framebuffers_ = std::vector<VkFramebuffer>(swapchain_imagecount);

	//create the framebuffers for the swapchain images. This will connect the render-pass to the images for rendering
	VkFramebufferCreateInfo fwd_info = vkinit::framebuffer_create_info(renderPass_, windowExtent_);
	VkImageView attachments[2];
	attachments[0] = rawRenderImage_.defaultView_;
	attachments[1] = depthImage_.defaultView_;

	fwd_info.pAttachments = attachments;
	fwd_info.attachmentCount = 2;
	VK_CHECK(vkCreateFramebuffer(device_, &fwd_info, nullptr, &forwardFramebuffer_));

	//create the framebuffer for shadow pass	
	VkFramebufferCreateInfo sh_info = vkinit::framebuffer_create_info(shadowPass_, shadowExtent_);
	sh_info.pAttachments = &shadowImage_.defaultView_;
	sh_info.attachmentCount = 1;
	VK_CHECK(vkCreateFramebuffer(device_, &sh_info, nullptr, &shadowFramebuffer_));
	
	for (uint32_t i = 0; i < swapchain_imagecount; i++) {

		//create the framebuffers for the swapchain images. This will connect the render-pass to the images for rendering
		VkFramebufferCreateInfo fb_info = vkinit::framebuffer_create_info(copyPass_, windowExtent_);
		fb_info.pAttachments = &swapchainImageViews_[i];
		fb_info.attachmentCount = 1;
		VK_CHECK(vkCreateFramebuffer(device_, &fb_info, nullptr, &framebuffers_[i]));

		mainDeletionQueue_.push_function([=]() {
			vkDestroyFramebuffer(device_, framebuffers_[i], nullptr);
			vkDestroyImageView(device_, swapchainImageViews_[i], nullptr);
			});
	}
}

void VulkanEngine::init_commands()
{
	//create a command pool for commands submitted to the graphics queue.
	//we also want the pool to allow for resetting of individual command buffers
	VkCommandPoolCreateInfo commandPoolInfo = vkinit::command_pool_create_info(graphicsQueueFamily_, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);


	for (int i = 0; i < FRAME_OVERLAP; i++) {


		VK_CHECK(vkCreateCommandPool(device_, &commandPoolInfo, nullptr, &frames_[i].commandPool_));

		//allocate the default command buffer that we will use for rendering
		VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(frames_[i].commandPool_, 1);

		VK_CHECK(vkAllocateCommandBuffers(device_, &cmdAllocInfo, &frames_[i].mainCommandBuffer_));

		mainDeletionQueue_.push_function([=]() {
			vkDestroyCommandPool(device_, frames_[i].commandPool_, nullptr);
		});

		
	}
	graphicsQueueContext_ = TracyVkContext(physicalDevice_, device_, graphicsQueue_, frames_[0].mainCommandBuffer_);

	
	VkCommandPoolCreateInfo uploadCommandPoolInfo = vkinit::command_pool_create_info(graphicsQueueFamily_);
	//create pool for upload context
	VK_CHECK(vkCreateCommandPool(device_, &uploadCommandPoolInfo, nullptr, &uploadContext_.commandPool_));

	mainDeletionQueue_.push_function([=]() {
		vkDestroyCommandPool(device_, uploadContext_.commandPool_, nullptr);
		});
}

void VulkanEngine::init_sync_structures()
{
	//create syncronization structures
	//one fence to control when the gpu has finished rendering the frame,
	//and 2 semaphores to syncronize rendering with swapchain
	//we want the fence to start signalled so we can wait on it on the first frame
	VkFenceCreateInfo fenceCreateInfo = vkinit::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);

	VkSemaphoreCreateInfo semaphoreCreateInfo = vkinit::semaphore_create_info();

	for (int i = 0; i < FRAME_OVERLAP; i++) {

		VK_CHECK(vkCreateFence(device_, &fenceCreateInfo, nullptr, &frames_[i].renderFence_));

		//enqueue the destruction of the fence
		mainDeletionQueue_.push_function([=]() {
			vkDestroyFence(device_, frames_[i].renderFence_, nullptr);
			});


		VK_CHECK(vkCreateSemaphore(device_, &semaphoreCreateInfo, nullptr, &frames_[i].presentSemaphore_));
		VK_CHECK(vkCreateSemaphore(device_, &semaphoreCreateInfo, nullptr, &frames_[i].renderSemaphore_));

		//enqueue the destruction of semaphores
		mainDeletionQueue_.push_function([=]() {
			vkDestroySemaphore(device_, frames_[i].presentSemaphore_, nullptr);
			vkDestroySemaphore(device_, frames_[i].renderSemaphore_, nullptr);
			});
	}


	VkFenceCreateInfo uploadFenceCreateInfo = vkinit::fence_create_info();

	VK_CHECK(vkCreateFence(device_, &uploadFenceCreateInfo, nullptr, &uploadContext_.uploadFence_));
	mainDeletionQueue_.push_function([=]() {
		vkDestroyFence(device_, uploadContext_.uploadFence_, nullptr);
		});
}


void VulkanEngine::init_pipelines()
{	
	materialSystem_ = new vkutil::MaterialSystem();
	materialSystem_->init(this);
	materialSystem_->build_default_templates();
		
	//fullscreen triangle pipeline for blits
	ShaderEffect* blitEffect = new ShaderEffect();
	blitEffect->add_stage(shaderCache_.get_shader(shader_path("fullscreen.vert.spv")), VK_SHADER_STAGE_VERTEX_BIT);
	blitEffect->add_stage(shaderCache_.get_shader(shader_path("blit.frag.spv")), VK_SHADER_STAGE_FRAGMENT_BIT);
	blitEffect->reflect_layout(device_, nullptr, 0);


	PipelineBuilder pipelineBuilder;

	//input assembly is the configuration for drawing triangle lists, strips, or individual points.
	//we are just going to draw triangle list
	pipelineBuilder.inputAssembly_ = vkinit::input_assembly_create_info(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

	//configure the rasterizer to draw filled triangles
	pipelineBuilder.rasterizer_ = vkinit::rasterization_state_create_info(VK_POLYGON_MODE_FILL);
	pipelineBuilder.rasterizer_.cullMode = VK_CULL_MODE_NONE;
	//we dont use multisampling, so just run the default one
	pipelineBuilder.multisampling_ = vkinit::multisampling_state_create_info();

	//a single blend attachment with no blending and writing to RGBA
	pipelineBuilder.colorBlendAttachment_ = vkinit::color_blend_attachment_state();


	//default depthtesting
	pipelineBuilder.depthStencil_ = vkinit::depth_stencil_create_info(true, true, VK_COMPARE_OP_GREATER_OR_EQUAL);

	//build blit pipeline
	pipelineBuilder.setShaders(blitEffect);

	//blit pipeline uses hardcoded triangle so no need for vertex input
	pipelineBuilder.clear_vertex_input();

	pipelineBuilder.depthStencil_ = vkinit::depth_stencil_create_info(false, false, VK_COMPARE_OP_ALWAYS);

	blitPipeline_ = pipelineBuilder.build_pipeline(device_, copyPass_);
	blitLayout_ = blitEffect->builtLayout;
	
	mainDeletionQueue_.push_function([=]() {
		//vkDestroyPipeline(_device, meshPipeline, nullptr);
		vkDestroyPipeline(device_, blitPipeline_, nullptr);
	});


	//load the compute shaders
	load_compute_shader(shader_path("indirect_cull.comp.spv").c_str(), cullPipeline_, cullLayout_);

	load_compute_shader(shader_path("depthReduce.comp.spv").c_str(), depthReducePipeline_, depthReduceLayout_);

	load_compute_shader(shader_path("sparse_upload.comp.spv").c_str(), sparseUploadPipeline_, sparseUploadLayout_);
}

bool VulkanEngine::load_compute_shader(const char* shaderPath, VkPipeline& pipeline, VkPipelineLayout& layout)
{
	ShaderModule computeModule;
	if (!vkutil::load_shader_module(device_, shaderPath, &computeModule))

	{
		std::cout << "Error when building compute shader shader module" << std::endl;
		return false;
	}

	ShaderEffect* computeEffect = new ShaderEffect();;
	computeEffect->add_stage(&computeModule, VK_SHADER_STAGE_COMPUTE_BIT);

	computeEffect->reflect_layout(device_, nullptr, 0);

	ComputePipelineBuilder computeBuilder;
	computeBuilder.pipelineLayout_ = computeEffect->builtLayout;
	computeBuilder.shaderStage_ = vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_COMPUTE_BIT, computeModule.module);


	layout = computeEffect->builtLayout;
	pipeline = computeBuilder.build_pipeline(device_);

	vkDestroyShaderModule(device_, computeModule.module, nullptr);

	mainDeletionQueue_.push_function([=]() {
		vkDestroyPipeline(device_, pipeline, nullptr);

		vkDestroyPipelineLayout(device_, layout, nullptr);
	});

	return true;
}




void VulkanEngine::load_meshes()
{
	Mesh triMesh{};
	triMesh.bounds_.valid = false;
	//make the array 3 vertices long
	triMesh.vertices_.resize(3);

	//vertex positions
	triMesh.vertices_[0].position = { 1.f,1.f, 0.0f };
	triMesh.vertices_[1].position = { -1.f,1.f, 0.0f };
	triMesh.vertices_[2].position = { 0.f,-1.f, 0.0f };

	//vertex colors, all green
	triMesh.vertices_[0].color = { 0.f,1.f, 0.0f }; //pure green
	triMesh.vertices_[1].color = { 0.f,1.f, 0.0f }; //pure green
	triMesh.vertices_[2].color = { 0.f,1.f, 0.0f }; //pure green
	//we dont care about the vertex normals
	upload_mesh(triMesh);
	meshes_["triangle"] = triMesh;
}


void VulkanEngine::load_images()
{
	load_image_to_cache("white", asset_path("Sponza/white.tx").c_str());
}


bool VulkanEngine::load_image_to_cache(const char* name, const char* path)
{
	ZoneScopedNC("Load Texture", tracy::Color::Yellow);
	Texture newtex;

	if (loadedTextures_.find(name) != loadedTextures_.end()) return true;

	bool result = vkutil::load_image_from_asset(*this, path, newtex.image);

	if (!result)
	{
		LOG_ERROR("Error When texture {} at path {}", name, path);
		return false;
	}
	else {
		LOG_SUCCESS("Loaded texture {} at path {}", name, path);
	}
	newtex.imageView = newtex.image.defaultView_;
	//VkImageViewCreateInfo imageinfo = vkinit::imageview_create_info(VK_FORMAT_R8G8B8A8_UNORM, newtex.image._image, VK_IMAGE_ASPECT_COLOR_BIT);
	//imageinfo.subresourceRange.levelCount = newtex.image.mipLevels;
	//vkCreateImageView(_device, &imageinfo, nullptr, &newtex.imageView);

	loadedTextures_[name] = newtex;
	return true;
}

void VulkanEngine::upload_mesh(Mesh& mesh)
{
	ZoneScopedNC("Upload Mesh", tracy::Color::Orange);


	const size_t vertex_buffer_size = mesh.vertices_.size() * sizeof(Vertex);
	const size_t index_buffer_size = mesh.indices_.size() * sizeof(uint32_t);
	const size_t bufferSize = vertex_buffer_size + index_buffer_size;
	//allocate vertex buffer
	VkBufferCreateInfo vertexBufferInfo = {};
	vertexBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	vertexBufferInfo.pNext = nullptr;
	//this is the total size, in bytes, of the buffer we are allocating
	vertexBufferInfo.size = vertex_buffer_size;
	//this buffer is going to be used as a Vertex Buffer
	vertexBufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

	//allocate vertex buffer
	VkBufferCreateInfo indexBufferInfo = {};
	indexBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	indexBufferInfo.pNext = nullptr;
	//this is the total size, in bytes, of the buffer we are allocating
	indexBufferInfo.size = index_buffer_size;
	//this buffer is going to be used as a Vertex Buffer
	indexBufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

	//let the VMA library know that this data should be writeable by CPU, but also readable by GPU
	VmaAllocationCreateInfo vmaallocInfo = {};
	vmaallocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;

	AllocatedBufferUntyped stagingBuffer;

	//allocate the buffer
	VK_CHECK(vmaCreateBuffer(allocator_, &vertexBufferInfo, &vmaallocInfo,
		&mesh.vertexBuffer_.buffer_,
		&mesh.vertexBuffer_.allocation_,
		nullptr));
	//copy vertex data
	char* data;
	vmaMapMemory(allocator_, mesh.vertexBuffer_.allocation_, (void**)&data);

	memcpy(data, mesh.vertices_.data(), vertex_buffer_size);

	vmaUnmapMemory(allocator_, mesh.vertexBuffer_.allocation_);

	if (index_buffer_size != 0)
	{
		//allocate the buffer
		VK_CHECK(vmaCreateBuffer(allocator_, &indexBufferInfo, &vmaallocInfo,
			&mesh.indexBuffer_.buffer_,
			&mesh.indexBuffer_.allocation_,
			nullptr));
		vmaMapMemory(allocator_, mesh.indexBuffer_.allocation_, (void**)&data);

		memcpy(data, mesh.indices_.data(), index_buffer_size);

		vmaUnmapMemory(allocator_, mesh.indexBuffer_.allocation_);
	}
}

Mesh* VulkanEngine::get_mesh(const std::string& name)
{
	auto it = meshes_.find(name);
	if (it == meshes_.end()) {
		return nullptr;
	}
	else {
		return &(*it).second;
	}
}

void VulkanEngine::init_scene()
{
	VkSamplerCreateInfo samplerInfo = vkinit::sampler_create_info(VK_FILTER_NEAREST);

	VkSampler blockySampler;
	vkCreateSampler(device_, &samplerInfo, nullptr, &blockySampler);

	samplerInfo = vkinit::sampler_create_info(VK_FILTER_LINEAR);

	samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	//info.anisotropyEnable = true;
	samplerInfo.mipLodBias = 2;
	samplerInfo.maxLod = 30.f;
	samplerInfo.minLod = 3;
	VkSampler smoothSampler;

	vkCreateSampler(device_, &samplerInfo, nullptr, &smoothSampler);
	

	{
		vkutil::MaterialData texturedInfo;
		texturedInfo.baseTemplate = "texturedPBR_opaque";
		texturedInfo.parameters = nullptr;

		vkutil::SampledTexture whiteTex;
		whiteTex.sampler = smoothSampler;
		whiteTex.view = loadedTextures_["white"].imageView;

		texturedInfo.textures.push_back(whiteTex);

		vkutil::Material* newmat = materialSystem_->build_material("textured", texturedInfo);
	}
	{
		vkutil::MaterialData matinfo;
		matinfo.baseTemplate = "texturedPBR_opaque";
		matinfo.parameters = nullptr;
	
		vkutil::SampledTexture whiteTex;
		whiteTex.sampler = smoothSampler;
		whiteTex.view = loadedTextures_["white"].imageView;

		matinfo.textures.push_back(whiteTex);

		vkutil::Material* newmat = materialSystem_->build_material("default", matinfo);

	}


	
	glm::mat4 translation = glm::translate(glm::mat4{ 1.0 }, glm::vec3(1000, 0, 0));
	glm::mat4 scale = glm::scale(glm::mat4{ 1.0 }, glm::vec3(300));

	load_prefab(asset_path("FlightHelmet/FlightHelmet.pfb").c_str(), (translation * scale));

	translation = glm::translate(glm::mat4{ 1.0 }, glm::vec3(500, 100, 0));
	scale = glm::scale(glm::mat4{ 1.0 }, glm::vec3(100));

	load_prefab(asset_path("DamagedHelmet/DamagedHelmet.pfb").c_str(), (translation * scale));



// 	int dimHelmets =1;
// 	for (int x = -dimHelmets; x <= dimHelmets; x++) {
// 		for (int y = -dimHelmets; y <= dimHelmets; y++) {
// 	
// 			glm::mat4 translation = glm::translate(glm::mat4{ 1.0 }, glm::vec3(x * 5, 10, y * 5));
// 			glm::mat4 scale = glm::scale(glm::mat4{ 1.0 }, glm::vec3(10));
// 	
// 			load_prefab(asset_path("FlightHelmet/FlightHelmet.pfb").c_str(),(translation * scale));
// 		}
// 	}

	glm::mat4 sponzaMatrix = glm::scale(glm::mat4{ 1.0 }, glm::vec3(1));;
	
	glm::mat4 unrealFixRotation = glm::rotate(glm::radians(-90.f), glm::vec3{ 1,0,0 });
	
	load_prefab(asset_path("Sponza/Sponza.pfb").c_str(), sponzaMatrix);
// 	load_prefab(asset_path("scifi/TopDownScifi.pfb").c_str(),  glm::translate(glm::vec3{0,20,0}));
// 	int dimcities = 2;
// 	for (int x = -dimcities; x <= dimcities; x++) {
// 		for (int y = -dimcities; y <= dimcities; y++) {
// 	
// 			glm::mat4 translation = glm::translate(glm::mat4{ 1.0 }, glm::vec3(x * 300, y, y * 300));
// 			glm::mat4 scale = glm::scale(glm::mat4{ 1.0 }, glm::vec3(10));
// 	
// 			
// 			glm::mat4 cityMatrix = translation;// * glm::scale(glm::mat4{ 1.0f }, glm::vec3(.01f));
// 			//load_prefab(asset_path("scifi/TopDownScifi.pfb").c_str(), unrealFixRotation * glm::scale(glm::mat4{ 1.0 }, glm::vec3(.01)));
// 			//load_prefab(asset_path("PolyCity/PolyCity.pfb").c_str(), cityMatrix);
// 			load_prefab(asset_path("CITY/polycity.pfb").c_str(), cityMatrix);
// 		//	load_prefab(asset_path("scifi/TopDownScifi.pfb").c_str(), cityMatrix);
// 		}
// 	}
	

	//for (int x = -20; x <= 20; x++) {
	//	for (int y = -20; y <= 20; y++) {
	//
	//		RenderObject tri;
	//		tri.mesh = get_mesh("triangle");
	//		tri.material = get_material("defaultmesh");
	//		glm::mat4 translation = glm::translate(glm::mat4{ 1.0 }, glm::vec3(x, 0, y));
	//		glm::mat4 scale = glm::scale(glm::mat4{ 1.0 }, glm::vec3(0.2, 0.2, 0.2));
	//		tri.transformMatrix = translation * scale;
	//
	//		refresh_renderbounds(&tri);
	//		_renderScene.register_object(&tri, PassTypeFlags::Forward);
	//	}
	//}
}


void VulkanEngine::ready_cull_data(RenderScene::MeshPass& pass, VkCommandBuffer cmd)
{
	//copy from the cleared indirect buffer into the one we will use on rendering. This one happens every frame
	VkBufferCopy indirectCopy;
	indirectCopy.dstOffset = 0;
	indirectCopy.size = pass.batches_.size() * sizeof(GPUIndirectObject);
	indirectCopy.srcOffset = 0;
	vkCmdCopyBuffer(cmd, pass.clearIndirectBuffer_.buffer_, pass.drawIndirectBuffer_.buffer_, 1, &indirectCopy);

	{
		VkBufferMemoryBarrier barrier = vkinit::buffer_barrier(pass.drawIndirectBuffer_.buffer_, graphicsQueueFamily_);
		barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

		cullReadyBarriers_.push_back(barrier);
		//vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
	}
}

AllocatedBufferUntyped VulkanEngine::create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage, VkMemoryPropertyFlags required_flags)
{
	//allocate vertex buffer
	VkBufferCreateInfo bufferInfo = {};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.pNext = nullptr;
	bufferInfo.size = allocSize;

	bufferInfo.usage = usage;


	//let the VMA library know that this data should be writeable by CPU, but also readable by GPU
	VmaAllocationCreateInfo vmaallocInfo = {};
	vmaallocInfo.usage = memoryUsage;
	vmaallocInfo.requiredFlags = required_flags;
	AllocatedBufferUntyped newBuffer;

	//allocate the buffer
	VK_CHECK(vmaCreateBuffer(allocator_, &bufferInfo, &vmaallocInfo,
		&newBuffer.buffer_,
		&newBuffer.allocation_,
		nullptr));
	newBuffer.size_ = allocSize;
	return newBuffer;
}


void VulkanEngine::reallocate_buffer(AllocatedBufferUntyped& buffer, size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage, VkMemoryPropertyFlags required_flags /*= 0*/)
{
	AllocatedBufferUntyped newBuffer = create_buffer(allocSize, usage, memoryUsage, required_flags);

	get_current_frame().frameDeletionQueue_.push_function([=]() {

		vmaDestroyBuffer(allocator_, buffer.buffer_, buffer.allocation_);
	});

	buffer = newBuffer;
}

size_t VulkanEngine::pad_uniform_buffer_size(size_t originalSize)
{
	// Calculate required alignment based on minimum device offset alignment
	size_t minUboAlignment = gpuProperties_.limits.minUniformBufferOffsetAlignment;
	size_t alignedSize = originalSize;
	if (minUboAlignment > 0) {
		alignedSize = (alignedSize + minUboAlignment - 1) & ~(minUboAlignment - 1);
	}
	return alignedSize;
}


void VulkanEngine::immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function)
{
	
	ZoneScopedNC("Inmediate Submit", tracy::Color::White);

	VkCommandBuffer cmd;

	//allocate the default command buffer that we will use for rendering
	VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(uploadContext_.commandPool_, 1);

	VK_CHECK(vkAllocateCommandBuffers(device_, &cmdAllocInfo, &cmd));

	//begin the command buffer recording. We will use this command buffer exactly once, so we want to let vulkan know that
	VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

	VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

	function(cmd);
	

	VK_CHECK(vkEndCommandBuffer(cmd));

	VkSubmitInfo submit = vkinit::submit_info(&cmd);


	//submit command buffer to the queue and execute it.
	// _renderFence will now block until the graphic commands finish execution
	VK_CHECK(vkQueueSubmit(graphicsQueue_, 1, &submit, uploadContext_.uploadFence_));

	vkWaitForFences(device_, 1, &uploadContext_.uploadFence_, true, 9999999999);
	vkResetFences(device_, 1, &uploadContext_.uploadFence_);

	vkResetCommandPool(device_, uploadContext_.commandPool_, 0);
}


bool VulkanEngine::load_prefab(const char* path, glm::mat4 root)
{
	int rng = rand();
	
	ZoneScopedNC("Load Prefab", tracy::Color::Red);

	auto pf = prefabCache_.find(path);
	if (pf == prefabCache_.end())
	{
		assets::AssetFile file;
		bool loaded = assets::load_binaryfile(path, file);

		if (!loaded) {
			LOG_FATAL("Error When loading prefab file at path {}",path);
			return false;
		}
		else {
			LOG_SUCCESS("Prefab {} loaded to cache", path);
		}

		prefabCache_[path] = new assets::PrefabInfo;

		*prefabCache_[path] = assets::read_prefab_info(&file);
	}

	assets::PrefabInfo* prefab = prefabCache_[path];

	VkSamplerCreateInfo samplerInfo = vkinit::sampler_create_info(VK_FILTER_LINEAR);
	samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;


	VkSampler smoothSampler;
	vkCreateSampler(device_, &samplerInfo, nullptr, &smoothSampler);


	std::unordered_map<uint64_t, glm::mat4> node_worldmats;

	std::vector<std::pair<uint64_t, glm::mat4>> pending_nodes;
	for (auto& [k, v] : prefab->node_matrices)
	{
		
		glm::mat4 nodematrix{ 1.f };

		auto nm = prefab->matrices[v];
		memcpy(&nodematrix, &nm, sizeof(glm::mat4));

		//check if it has parents
		auto matrixIT = prefab->node_parents.find(k);
		if (matrixIT == prefab->node_parents.end()) {
			//add to worldmats 
			node_worldmats[k] = root* nodematrix;
		}
		else {
			//enqueue
			pending_nodes.push_back({ k,nodematrix });
		}
	}

	//process pending nodes list until it empties
	while (pending_nodes.size() > 0)
	{
		for (int i = 0; i < pending_nodes.size(); i++)
		{
			uint64_t node = pending_nodes[i].first;
			uint64_t parent = prefab->node_parents[node];

			//try to find parent in cache
			auto matrixIT = node_worldmats.find(parent);
			if (matrixIT != node_worldmats.end()) {

				//transform with the parent
				glm::mat4 nodematrix = (matrixIT)->second * pending_nodes[i].second;

				node_worldmats[node] = nodematrix;

				//remove from queue, pop last
				pending_nodes[i] = pending_nodes.back();
				pending_nodes.pop_back();
				i--;
			}
		}
		
	}

	std::vector<MeshObject> prefab_renderables;
	prefab_renderables.reserve(prefab->node_meshes.size());

	for (auto& [k, v] : prefab->node_meshes)
	{
		
		//load mesh

		if (v.mesh_path.find("Sky") != std::string::npos) {
			continue;
		}

		if (!get_mesh(v.mesh_path.c_str()))
		{
			Mesh mesh{};
			mesh.load_from_meshasset(asset_path(v.mesh_path).c_str());

			upload_mesh(mesh);

			meshes_[v.mesh_path.c_str()] = mesh;
		}

		
		auto materialName = v.material_path.c_str();
		//load material
		
		vkutil::Material* objectMaterial = materialSystem_->get_material(materialName);
		if (!objectMaterial)
		{
			assets::AssetFile materialFile;
			bool loaded = assets::load_binaryfile(asset_path(materialName).c_str(), materialFile);
			
			if (loaded)
			{
				assets::MaterialInfo material = assets::read_material_info(&materialFile);

				auto texture = material.textures["baseColor"];
				if (texture.size() <= 3)
				{
					texture = "Sponza/white.tx";
				}

				loaded = load_image_to_cache(texture.c_str(), asset_path(texture).c_str());
				
				if (loaded)
				{
					vkutil::SampledTexture tex;
					tex.view = loadedTextures_[texture].imageView;
					tex.sampler = smoothSampler;

					vkutil::MaterialData info;
					info.parameters = nullptr;

					if (material.transparency == assets::TransparencyMode::Transparent)
					{
						info.baseTemplate = "texturedPBR_transparent";
					}
					else {
						info.baseTemplate = "texturedPBR_opaque";
					}
					
					info.textures.push_back(tex);

					objectMaterial = materialSystem_->build_material(materialName, info);

					if (!objectMaterial)
					{
						LOG_ERROR("Error When building material {}", v.material_path);
					}
				}
				else
				{
					LOG_ERROR("Error When loading image at {}", v.material_path);
				}
			}
			else
			{
				LOG_ERROR("Error When loading material at path {}", v.material_path);
			}
		}
		
		MeshObject loadmesh;
		//transparent objects will be invisible
		
		loadmesh.bDrawForwardPass = true;
		loadmesh.bDrawShadowPass = true;
		

		glm::mat4 nodematrix{ 1.f };

		auto matrixIT = node_worldmats.find(k);
		if (matrixIT != node_worldmats.end()) {
			auto nm = (*matrixIT).second;
			memcpy(&nodematrix, &nm, sizeof(glm::mat4));
		}		
		
		loadmesh.mesh = get_mesh(v.mesh_path.c_str());
		loadmesh.transformMatrix = nodematrix;
		loadmesh.material = objectMaterial;

		refresh_renderbounds(&loadmesh);

		//sort key from location
		int32_t lx = int(loadmesh.bounds.origin.x / 10.f);
		int32_t ly = int(loadmesh.bounds.origin.y / 10.f);

		uint32_t key =  uint32_t(std::hash<int32_t>()(lx) ^ std::hash<int32_t>()(ly^1337));

		loadmesh.customSortKey = 0;// rng;// key;
		

		prefab_renderables.push_back(loadmesh);
		//_renderables.push_back(loadmesh);
	}

	renderScene_.register_object_batch(prefab_renderables.data(), static_cast<uint32_t>(prefab_renderables.size()));



	return true;
}


std::string VulkanEngine::asset_path(std::string_view path)
{
	return "../../../assets_export/" + std::string(path);
}



std::string VulkanEngine::shader_path(std::string_view path)
{
	return "../shaders/" + std::string(path);
}

void VulkanEngine::refresh_renderbounds(MeshObject* object)
{
	//dont try to update invalid bounds
	if (!object->mesh->bounds_.valid) return;

	RenderBounds originalBounds = object->mesh->bounds_;

	//convert bounds to 8 vertices, and transform those
	std::array<glm::vec3, 8> boundsVerts;

	for (int i = 0; i < 8; i++) {
		boundsVerts[i] = originalBounds.origin;
	}

	boundsVerts[0] += originalBounds.extents * glm::vec3(1, 1, 1);
	boundsVerts[1] += originalBounds.extents * glm::vec3(1, 1, -1);
	boundsVerts[2] += originalBounds.extents * glm::vec3(1, -1, 1);
	boundsVerts[3] += originalBounds.extents * glm::vec3(1, -1, -1);
	boundsVerts[4] += originalBounds.extents * glm::vec3(-1, 1, 1);
	boundsVerts[5] += originalBounds.extents * glm::vec3(-1, 1, -1);
	boundsVerts[6] += originalBounds.extents * glm::vec3(-1, -1, 1);
	boundsVerts[7] += originalBounds.extents * glm::vec3(-1, -1, -1);
	
	//recalc max/min
	glm::vec3 min{ std::numeric_limits<float>().max() };
	glm::vec3 max{ -std::numeric_limits<float>().max() };

	glm::mat4 m = object->transformMatrix;

	//transform every vertex, accumulating max/min
	for (int i = 0; i < 8; i++) {
		boundsVerts[i] = m * glm::vec4(boundsVerts[i],1.f);

		min = glm::min(boundsVerts[i], min);
		max = glm::max(boundsVerts[i], max);
	}

	glm::vec3 extents = (max - min) / 2.f;
	glm::vec3 origin = min + extents;

	float max_scale = 0;
	max_scale = std::max( glm::length(glm::vec3(m[0][0], m[0][1], m[0][2])),max_scale);
	max_scale = std::max( glm::length(glm::vec3(m[1][0], m[1][1], m[1][2])),max_scale);
	max_scale = std::max( glm::length(glm::vec3(m[2][0], m[2][1], m[2][2])),max_scale);

	float radius = max_scale * originalBounds.radius;


	object->bounds.extents = extents;
	object->bounds.origin = origin;
	object->bounds.radius = radius;
	object->bounds.valid = true;
}


void VulkanEngine::unmap_buffer(AllocatedBufferUntyped& buffer)
{
	vmaUnmapMemory(allocator_, buffer.allocation_);
}

void VulkanEngine::init_descriptors()
{
	descriptorAllocator_ = new vkutil::DescriptorAllocator{};
	descriptorAllocator_->init(device_);

	descriptorLayoutCache_ = new vkutil::DescriptorLayoutCache{};
	descriptorLayoutCache_->init(device_);


// 	VkDescriptorSetLayoutBinding textureBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0);
// 
// 	VkDescriptorSetLayoutCreateInfo set3info = {};
// 	set3info.bindingCount = 1;
// 	set3info.flags = 0;
// 	set3info.pNext = nullptr;
// 	set3info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
// 	set3info.pBindings = &textureBind;
// 
// 	singleTextureSetLayout_ = descriptorLayoutCache_->create_descriptor_layout(&set3info);


	const size_t sceneParamBufferSize = FRAME_OVERLAP * pad_uniform_buffer_size(sizeof(GPUSceneData));


	for (int i = 0; i < FRAME_OVERLAP; i++)
	{
		frames_[i].dynamicDescriptorAllocator_ = new vkutil::DescriptorAllocator{};
		frames_[i].dynamicDescriptorAllocator_->init(device_);

		//1 megabyte of dynamic data buffer
		auto dynamicDataBuffer = create_buffer(1000000, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
		frames_[i].dynamicData_.init(allocator_, dynamicDataBuffer, gpuProperties_.limits.minUniformBufferOffsetAlignment); 

		//20 megabyte of debug output
		frames_[i].debugOutputBuffer_ = create_buffer(200000000, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);
	}
}

void VulkanEngine::init_imgui()
{
	//1: create descriptor pool for IMGUI
	// the size of the pool is very oversize, but its copied from imgui demo itself.
	VkDescriptorPoolSize pool_sizes[] =
	{
		{ VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
		{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
		{ VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
		{ VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
	};

	VkDescriptorPoolCreateInfo pool_info = {};
	pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
	pool_info.maxSets = 1000;
	pool_info.poolSizeCount = 11;// std::size(pool_sizes);
	pool_info.pPoolSizes = pool_sizes;

	VkDescriptorPool imguiPool;
	VK_CHECK(vkCreateDescriptorPool(device_, &pool_info, nullptr, &imguiPool));


	// 2: initialize imgui library

	//this initializes the core structures of imgui
	ImGui::CreateContext();
	ImGui::GetIO().IniFilename = NULL;

	//this initializes imgui for SDL
	ImGui_ImplSDL2_InitForVulkan(window_);

	//this initializes imgui for Vulkan
	ImGui_ImplVulkan_InitInfo init_info = {};
	init_info.Instance = instance_;
	init_info.PhysicalDevice = physicalDevice_;
	init_info.Device = device_;
	init_info.Queue = graphicsQueue_;
	init_info.DescriptorPool = imguiPool;
	init_info.MinImageCount = 3;
	init_info.ImageCount = 3;

	ImGui_ImplVulkan_Init(&init_info, renderPass_);

	//execute a gpu command to upload imgui font textures
	immediate_submit([&](VkCommandBuffer cmd) {
		ImGui_ImplVulkan_CreateFontsTexture(cmd);
		});

	//clear font textures from cpu data
	ImGui_ImplVulkan_DestroyFontUploadObjects();

	//add the destroy the imgui created structures
	mainDeletionQueue_.push_function([=]() {

		vkDestroyDescriptorPool(device_, imguiPool, nullptr);
		ImGui_ImplVulkan_Shutdown();
		});
}


glm::mat4 DirectionalLight::get_projection()
{
	glm::mat4 projection = glm::orthoLH_ZO(-shadowExtent.x, shadowExtent.x, -shadowExtent.y, shadowExtent.y, -shadowExtent.z, shadowExtent.z);
	return projection;
}

glm::mat4 DirectionalLight::get_view()
{
	glm::vec3 camPos = lightPosition;

	glm::vec3 camFwd = lightDirection;

	glm::mat4 view = glm::lookAt(camPos, camPos + camFwd, glm::vec3(1, 0, 0));
	return view;
}
