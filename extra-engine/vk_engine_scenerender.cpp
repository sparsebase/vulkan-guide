#include "vk_engine.h"

#include <SDL.h>
#include <SDL_vulkan.h>

#include <vk_types.h>
#include <vk_initializers.h>
#include <vk_descriptors.h>

#include "vk_textures.h"
#include "vk_shaders.h"

#include "Tracy.hpp"
#include "TracyVulkan.hpp"
#include "vk_profiler.h"
#include "cvars.h"

AutoCVar_Int CVAR_FreezeCull("culling.freeze", "Locks culling", 0, CVarFlags::EditCheckbox);

AutoCVar_Int CVAR_Shadowcast("gpu.shadowcast", "Use shadowcasting", 1, CVarFlags::EditCheckbox);

AutoCVar_Float CVAR_ShadowBias("gpu.shadowBias", "Distance cull", 5.25f);
AutoCVar_Float CVAR_SlopeBias("gpu.shadowBiasSlope", "Distance cull", 4.75f);


glm::vec4 normalizePlane(glm::vec4 p)
{
	return p / glm::length(glm::vec3(p));
}

void VulkanEngine::execute_compute_cull(VkCommandBuffer cmd, RenderScene::MeshPass& pass,CullParams& params )
{
	if (CVAR_FreezeCull.Get()) return;
	
	if (pass.batches_.size() == 0) return;
	TracyVkZone(graphicsQueueContext_, cmd, "Cull Dispatch");
	VkDescriptorBufferInfo objectBufferInfo = renderScene_.objectDataBuffer_.get_info();

	VkDescriptorBufferInfo dynamicInfo = get_current_frame().dynamicData_.source_.get_info();
	dynamicInfo.range = sizeof(GPUCameraData);

	VkDescriptorBufferInfo instanceInfo = pass.passObjectsBuffer_.get_info();

	VkDescriptorBufferInfo finalInfo = pass.compactedInstanceBuffer_.get_info();

	VkDescriptorBufferInfo indirectInfo = pass.drawIndirectBuffer_.get_info();

	VkDescriptorImageInfo depthPyramid;
	depthPyramid.sampler = depthSampler_;
	depthPyramid.imageView = depthPyramid_.defaultView_;
	depthPyramid.imageLayout = VK_IMAGE_LAYOUT_GENERAL;


	VkDescriptorSet COMPObjectDataSet;
	vkutil::DescriptorBuilder::begin(descriptorLayoutCache_, get_current_frame().dynamicDescriptorAllocator_)
		.bind_buffer(0, &objectBufferInfo, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
		.bind_buffer(1, &indirectInfo, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
		.bind_buffer(2, &instanceInfo, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
		.bind_buffer(3, &finalInfo, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
		.bind_image(4, &depthPyramid, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT)
		.bind_buffer(5, &dynamicInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
		.build(COMPObjectDataSet);


	glm::mat4 projection = params.projmat;
	glm::mat4 projectionT = transpose(projection);

	glm::vec4 frustumX = normalizePlane(projectionT[3] + projectionT[0]); // x + w < 0
	glm::vec4 frustumY = normalizePlane(projectionT[3] + projectionT[1]); // y + w < 0

	DrawCullData cullData = {};
	cullData.P00 = projection[0][0];
	cullData.P11 = projection[1][1];
	cullData.znear = 0.1f;
	cullData.zfar = params.drawDist;
	cullData.frustum[0] = frustumX.x;
	cullData.frustum[1] = frustumX.z;
	cullData.frustum[2] = frustumY.y;
	cullData.frustum[3] = frustumY.z;
	cullData.drawCount = static_cast<uint32_t>(pass.flat_batches_.size());
	cullData.cullingEnabled = params.frustrumCull;
	cullData.lodEnabled = false;
	cullData.occlusionEnabled = params.occlusionCull;
	cullData.lodBase = 10.f;
	cullData.lodStep = 1.5f;
	cullData.pyramidWidth = static_cast<float>(depthPyramidWidth_);
	cullData.pyramidHeight = static_cast<float>(depthPyramidHeight_);
	cullData.viewMat = params.viewmat;//get_view_matrix();

	cullData.AABBcheck = params.aabb;
	cullData.aabbmin_x = params.aabbmin.x;
	cullData.aabbmin_y = params.aabbmin.y;
	cullData.aabbmin_z = params.aabbmin.z;

	cullData.aabbmax_x = params.aabbmax.x;
	cullData.aabbmax_y = params.aabbmax.y;
	cullData.aabbmax_z = params.aabbmax.z;

	if (params.drawDist > 10000)
	{
		cullData.distanceCheck = false; 
	}
	else
	{
		cullData.distanceCheck = true;
	}

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cullPipeline_);

	vkCmdPushConstants(cmd, cullLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DrawCullData), &cullData);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cullLayout_, 0, 1, &COMPObjectDataSet, 0, nullptr);
	
	vkCmdDispatch(cmd, static_cast<uint32_t>((pass.flat_batches_.size() / 256)+1), 1, 1);


	//barrier the 2 buffers we just wrote for culling, the indirect draw one, and the instances one, so that they can be read well when rendering the pass
	{
		VkBufferMemoryBarrier barrier = vkinit::buffer_barrier(pass.compactedInstanceBuffer_.buffer_, graphicsQueueFamily_);
		barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;

		VkBufferMemoryBarrier barrier2 = vkinit::buffer_barrier(pass.drawIndirectBuffer_.buffer_, graphicsQueueFamily_);	
		barrier2.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		barrier2.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;

		VkBufferMemoryBarrier barriers[] = { barrier,barrier2 };

		postCullBarriers_.push_back(barrier);
		postCullBarriers_.push_back(barrier2);
		
	}
	if (*CVarSystem::Get()->GetIntCVar("culling.outputIndirectBufferToFile"))
	{
		uint32_t offset = get_current_frame().debugDataOffsets_.back();
		VkBufferCopy debugCopy;
		debugCopy.dstOffset = offset;
		debugCopy.size = pass.batches_.size() * sizeof(GPUIndirectObject);
		debugCopy.srcOffset = 0;
		vkCmdCopyBuffer(cmd, pass.drawIndirectBuffer_.buffer_, get_current_frame().debugOutputBuffer_.buffer_, 1, &debugCopy);
		get_current_frame().debugDataOffsets_.push_back(offset + static_cast<uint32_t>(debugCopy.size));
		get_current_frame().debugDataNames_.push_back("Cull Indirect Output");
	}
}
#include <future>
void VulkanEngine::ready_mesh_draw(VkCommandBuffer cmd)
{
	
	TracyVkZone(graphicsQueueContext_, get_current_frame().mainCommandBuffer_, "Data Refresh");
	ZoneScopedNC("Draw Upload", tracy::Color::Blue);

	//upload object data to gpu
	
	if (renderScene_.dirtyObjects_.size() > 0)
	{
		ZoneScopedNC("Refresh Object Buffer", tracy::Color::Red);

		size_t copySize = renderScene_.renderables_.size() * sizeof(GPUObjectData);
		if (renderScene_.objectDataBuffer_.size_ < copySize)
		{
			reallocate_buffer(renderScene_.objectDataBuffer_, copySize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
		}

		//if 80% of the objects are dirty, then just reupload the whole thing
		if (renderScene_.dirtyObjects_.size() >= renderScene_.renderables_.size() * 0.8)
		{
			AllocatedBuffer<GPUObjectData> newBuffer = create_buffer(copySize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

			GPUObjectData* objectSSBO = map_buffer(newBuffer);
			renderScene_.fill_objectData(objectSSBO);
			unmap_buffer(newBuffer);

			get_current_frame().frameDeletionQueue_.push_function([=]() {

				vmaDestroyBuffer(allocator_, newBuffer.buffer_, newBuffer.allocation_);
				});

		
			//copy from the uploaded cpu side instance buffer to the gpu one
			VkBufferCopy indirectCopy;
			indirectCopy.dstOffset = 0;
			indirectCopy.size = renderScene_.renderables_.size() * sizeof(GPUObjectData);
			indirectCopy.srcOffset = 0;
			vkCmdCopyBuffer(cmd, newBuffer.buffer_, renderScene_.objectDataBuffer_.buffer_, 1, &indirectCopy);
		}
		else {
			//update only the changed elements

			std::vector<VkBufferCopy> copies;
			copies.reserve(renderScene_.dirtyObjects_.size());

			uint64_t buffersize = sizeof(GPUObjectData) * renderScene_.dirtyObjects_.size();
			uint64_t vec4size = sizeof(glm::vec4);
			uint64_t intsize = sizeof(uint32_t);
			uint64_t wordsize = sizeof(GPUObjectData) / sizeof(uint32_t);
			uint64_t uploadSize = renderScene_.dirtyObjects_.size() * wordsize * intsize;
			AllocatedBuffer<GPUObjectData> newBuffer = create_buffer(buffersize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
			AllocatedBuffer<uint32_t> targetBuffer = create_buffer(uploadSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

			get_current_frame().frameDeletionQueue_.push_function([=]() {

				vmaDestroyBuffer(allocator_, newBuffer.buffer_, newBuffer.allocation_);
				vmaDestroyBuffer(allocator_, targetBuffer.buffer_, targetBuffer.allocation_);
				});

			uint32_t* targetData = map_buffer(targetBuffer);
			GPUObjectData* objectSSBO = map_buffer(newBuffer);
			uint32_t launchcount = static_cast<uint32_t>(renderScene_.dirtyObjects_.size() * wordsize);
			{
				ZoneScopedNC("Write dirty objects", tracy::Color::Red);
				uint32_t sidx = 0;
				for (int i = 0; i < renderScene_.dirtyObjects_.size(); i++)
				{
					renderScene_.write_object(objectSSBO + i, renderScene_.dirtyObjects_[i]);


					uint32_t dstOffset = static_cast<uint32_t>(wordsize * renderScene_.dirtyObjects_[i].handle);

					for (int b = 0; b < wordsize; b++ )
					{
						uint32_t tidx = dstOffset + b;
						targetData[sidx] = tidx;
						sidx++;
					}
				}
				launchcount = sidx;
			}
			unmap_buffer(newBuffer);
			unmap_buffer(targetBuffer); 
			
			VkDescriptorBufferInfo indexData = targetBuffer.get_info();

			VkDescriptorBufferInfo sourceData = newBuffer.get_info(); 

			VkDescriptorBufferInfo targetInfo = renderScene_.objectDataBuffer_.get_info();

			VkDescriptorSet COMPObjectDataSet;
			vkutil::DescriptorBuilder::begin(descriptorLayoutCache_, get_current_frame().dynamicDescriptorAllocator_)
				.bind_buffer(0, &indexData, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
				.bind_buffer(1, &sourceData, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
				.bind_buffer(2, &targetInfo, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
				.build(COMPObjectDataSet);

			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sparseUploadPipeline_);

			
			vkCmdPushConstants(cmd, sparseUploadLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &launchcount);

			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sparseUploadLayout_, 0, 1, &COMPObjectDataSet, 0, nullptr);

			vkCmdDispatch(cmd, ((launchcount) / 256) + 1, 1, 1);
		}

		VkBufferMemoryBarrier barrier = vkinit::buffer_barrier(renderScene_.objectDataBuffer_.buffer_, graphicsQueueFamily_);
		barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			
		uploadBarriers_.push_back(barrier);
		renderScene_.clear_dirty_objects();
	}

	RenderScene::MeshPass* passes[3] = { &renderScene_.forwardPass_,&renderScene_.transparentForwardPass_,&renderScene_.shadowPass_ };
	for (int p = 0; p < 3; p++)
	{
		auto& pass = *passes[p];


		//reallocate the gpu side buffers if needed

		if (pass.drawIndirectBuffer_.size_ < pass.batches_.size() * sizeof(GPUIndirectObject))
		{
			reallocate_buffer(pass.drawIndirectBuffer_, pass.batches_.size() * sizeof(GPUIndirectObject), VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
		}

		if (pass.compactedInstanceBuffer_.size_ < pass.flat_batches_.size() * sizeof(uint32_t))
		{
			reallocate_buffer(pass.compactedInstanceBuffer_, pass.flat_batches_.size() * sizeof(uint32_t), VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
		}

		if (pass.passObjectsBuffer_.size_ < pass.flat_batches_.size() * sizeof(GPUInstance))
		{
			reallocate_buffer(pass.passObjectsBuffer_, pass.flat_batches_.size() * sizeof(GPUInstance), VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
		}
	}

	std::vector<std::future<void>> async_calls;
	async_calls.reserve(9);

	std::vector<AllocatedBufferUntyped> unmaps;
	

	for (int p = 0; p < 3; p++)
	{
		RenderScene::MeshPass& pass = *passes[p];
		RenderScene::MeshPass* ppass = passes[p];

		RenderScene* pScene = &renderScene_;
		//if the pass has changed the batches, need to reupload them
		if (pass.needsIndirectRefresh_ && pass.batches_.size() > 0)
		{
			ZoneScopedNC("Refresh Indirect Buffer", tracy::Color::Red);

			AllocatedBuffer<GPUIndirectObject> newBuffer = create_buffer(sizeof(GPUIndirectObject) * pass.batches_.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

			GPUIndirectObject* indirect = map_buffer(newBuffer);

			

			async_calls.push_back(std::async(std::launch::async, [=] { 
				
				
				pScene->fill_indirectArray(indirect, *ppass);
				
			}));
			//async_calls.push_back([&]() {
			//	_renderScene.fill_indirectArray(indirect, pass);
			//});
			
			unmaps.push_back(newBuffer);

			//unmap_buffer(newBuffer);

			if (pass.clearIndirectBuffer_.buffer_ != VK_NULL_HANDLE)
			{
				AllocatedBufferUntyped deletionBuffer = pass.clearIndirectBuffer_;
				//add buffer to deletion queue of this frame
				get_current_frame().frameDeletionQueue_.push_function([=]() {

					vmaDestroyBuffer(allocator_, deletionBuffer.buffer_, deletionBuffer.allocation_);
				});
			}

			pass.clearIndirectBuffer_ = newBuffer;
			pass.needsIndirectRefresh_ = false;
		}

		if (pass.needsInstanceRefresh_ && pass.flat_batches_.size() >0)
		{
			ZoneScopedNC("Refresh Instancing Buffer", tracy::Color::Red);

			AllocatedBuffer<GPUInstance> newBuffer = create_buffer(sizeof(GPUInstance) * pass.flat_batches_.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

			GPUInstance* instanceData = map_buffer(newBuffer);
			async_calls.push_back(std::async(std::launch::async, [=] {


				pScene->fill_instancesArray(instanceData, *ppass);

			}));
			unmaps.push_back(newBuffer);
			//_renderScene.fill_instancesArray(instanceData, pass);
			//unmap_buffer(newBuffer);

			get_current_frame().frameDeletionQueue_.push_function([=]() {

				vmaDestroyBuffer(allocator_, newBuffer.buffer_, newBuffer.allocation_);
				});
			
			//copy from the uploaded cpu side instance buffer to the gpu one
			VkBufferCopy indirectCopy;
			indirectCopy.dstOffset = 0;
			indirectCopy.size = pass.flat_batches_.size() * sizeof(GPUInstance);
			indirectCopy.srcOffset = 0;
			vkCmdCopyBuffer(cmd, newBuffer.buffer_, pass.passObjectsBuffer_.buffer_, 1, &indirectCopy);

			VkBufferMemoryBarrier barrier = vkinit::buffer_barrier(pass.passObjectsBuffer_.buffer_, graphicsQueueFamily_);
			barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

			uploadBarriers_.push_back(barrier);

			pass.needsInstanceRefresh_ = false;
		}
	}

	for (auto& s : async_calls)
	{
		s.get();
	}
	for (auto b : unmaps)
	{
		unmap_buffer(b);
	}

	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, static_cast<uint32_t>(uploadBarriers_.size()), uploadBarriers_.data(), 0, nullptr);//1, &readBarrier);
	uploadBarriers_.clear();
}

void VulkanEngine::draw_objects_forward(VkCommandBuffer cmd, RenderScene::MeshPass& pass)
{
	ZoneScopedNC("DrawObjects", tracy::Color::Blue);
	//make a model view matrix for rendering the object
	//camera view
	glm::mat4 view = camera_.get_view_matrix();
	//camera projection
	glm::mat4 projection = camera_.get_projection_matrix();


	GPUCameraData camData;
	camData.proj = projection;
	camData.view = view;
	camData.viewproj = projection * view;

	sceneParameters_.sunlightShadowMatrix =mainLight_.get_projection() * mainLight_.get_view();


	float framed = (frameNumber_ / 120.f);
	sceneParameters_.ambientColor = glm::vec4{ 0.5 };
	sceneParameters_.sunlightColor = glm::vec4{ 1.f };
	sceneParameters_.sunlightDirection = glm::vec4(mainLight_.lightDirection * 1.f,1.f);

	sceneParameters_.sunlightColor.w = CVAR_Shadowcast.Get() ? 0 : 1;

	//push data to dynmem
	uint32_t scene_data_offset = get_current_frame().dynamicData_.push(sceneParameters_);

	uint32_t camera_data_offset = get_current_frame().dynamicData_.push(camData);
	

	VkDescriptorBufferInfo objectBufferInfo = renderScene_.objectDataBuffer_.get_info();

	VkDescriptorBufferInfo sceneInfo = get_current_frame().dynamicData_.source_.get_info();
	sceneInfo.range = sizeof(GPUSceneData);

	VkDescriptorBufferInfo camInfo = get_current_frame().dynamicData_.source_.get_info();
	camInfo.range = sizeof(GPUCameraData);

	VkDescriptorBufferInfo instanceInfo = pass.compactedInstanceBuffer_.get_info();


	VkDescriptorImageInfo shadowImage;
	shadowImage.sampler = shadowSampler_;

	shadowImage.imageView = shadowImage_.defaultView_;
	shadowImage.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

	VkDescriptorSet GlobalSet;
	vkutil::DescriptorBuilder::begin(descriptorLayoutCache_, get_current_frame().dynamicDescriptorAllocator_)
		.bind_buffer(0, &camInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, VK_SHADER_STAGE_VERTEX_BIT)
		.bind_buffer(1, &sceneInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, VK_SHADER_STAGE_VERTEX_BIT| VK_SHADER_STAGE_FRAGMENT_BIT)
		.bind_image(2, &shadowImage, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
		.build(GlobalSet);

	VkDescriptorSet ObjectDataSet;
	vkutil::DescriptorBuilder::begin(descriptorLayoutCache_, get_current_frame().dynamicDescriptorAllocator_)
		.bind_buffer(0, &objectBufferInfo, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT)
		.bind_buffer(1, &instanceInfo, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT)
		.build(ObjectDataSet);
	vkCmdSetDepthBias(cmd, 0, 0, 0);

	std::vector<uint32_t> dynamic_offsets;
	dynamic_offsets.push_back(camera_data_offset);
	dynamic_offsets.push_back(scene_data_offset);
	execute_draw_commands(cmd, pass, ObjectDataSet, dynamic_offsets, GlobalSet);
}


void VulkanEngine::execute_draw_commands(VkCommandBuffer cmd, RenderScene::MeshPass& pass, VkDescriptorSet ObjectDataSet, std::vector<uint32_t> dynamic_offsets, VkDescriptorSet GlobalSet)
{
	if(pass.batches_.size() > 0)
	{
		ZoneScopedNC("Draw Commit", tracy::Color::Blue4);
		Mesh* lastMesh = nullptr;
		VkPipeline lastPipeline{ VK_NULL_HANDLE };
		VkPipelineLayout lastLayout{ VK_NULL_HANDLE };
		VkDescriptorSet lastMaterialSet{ VK_NULL_HANDLE };

		VkDeviceSize offset = 0;
		vkCmdBindVertexBuffers(cmd, 0, 1, &renderScene_.mergedVertexBuffer_.buffer_, &offset);

		vkCmdBindIndexBuffer(cmd, renderScene_.mergedIndexBuffer_.buffer_, 0, VK_INDEX_TYPE_UINT32);

		stats.objects = static_cast<uint32_t>(pass.flat_batches_.size());
		for (int i = 0; i < pass.multibatches_.size(); i++)
		{
			auto& multibatch = pass.multibatches_[i];
			auto& instanceDraw = pass.batches_[multibatch.first];

			VkPipeline newPipeline = instanceDraw.material.shaderPass->pipeline;
			VkPipelineLayout newLayout = instanceDraw.material.shaderPass->layout;
			VkDescriptorSet newMaterialSet = instanceDraw.material.materialSet;

			Mesh* drawMesh = renderScene_.get_mesh(instanceDraw.meshID)->original;

			if (newPipeline != lastPipeline)
			{
				lastPipeline = newPipeline;
				vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, newPipeline);
				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, newLayout, 1, 1, &ObjectDataSet, 0, nullptr);

				//update dynamic binds
				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, newLayout, 0, 1, &GlobalSet, dynamic_offsets.size(), dynamic_offsets.data());
			}
			if (newMaterialSet != lastMaterialSet)
			{
				lastMaterialSet = newMaterialSet;
				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, newLayout, 2, 1, &newMaterialSet, 0, nullptr);
			}

			bool merged = renderScene_.get_mesh(instanceDraw.meshID)->isMerged;
			if (merged)
			{
				if (lastMesh != nullptr)
				{
					VkDeviceSize offset = 0;
					vkCmdBindVertexBuffers(cmd, 0, 1, &renderScene_.mergedVertexBuffer_.buffer_, &offset);

					vkCmdBindIndexBuffer(cmd, renderScene_.mergedIndexBuffer_.buffer_, 0, VK_INDEX_TYPE_UINT32);
					lastMesh = nullptr;
				}
			}
			else if (lastMesh != drawMesh) {

				//bind the mesh vertex buffer with offset 0
				VkDeviceSize offset = 0;
				vkCmdBindVertexBuffers(cmd, 0, 1, &drawMesh->vertexBuffer_.buffer_, &offset);

				if (drawMesh->indexBuffer_.buffer_ != VK_NULL_HANDLE) {
					vkCmdBindIndexBuffer(cmd, drawMesh->indexBuffer_.buffer_, 0, VK_INDEX_TYPE_UINT32);
				}
				lastMesh = drawMesh;
			}

			bool bHasIndices = drawMesh->indices_.size() > 0;
			if (!bHasIndices) {
				stats.draws++;
				stats.triangles += static_cast<int32_t>(drawMesh->vertices_.size() / 3) * instanceDraw.count;
				vkCmdDraw(cmd, static_cast<uint32_t>(drawMesh->vertices_.size()), instanceDraw.count, 0, instanceDraw.first);
			}
			else {
				stats.triangles += static_cast<int32_t>(drawMesh->indices_.size() / 3) * instanceDraw.count;

				vkCmdDrawIndexedIndirect(cmd, pass.drawIndirectBuffer_.buffer_, multibatch.first * sizeof(GPUIndirectObject), multibatch.count, sizeof(GPUIndirectObject));

				stats.draws++;
				stats.drawcalls += instanceDraw.count;
			}
		}
	}
}
void VulkanEngine::draw_objects_shadow(VkCommandBuffer cmd, RenderScene::MeshPass& pass)
{
	ZoneScopedNC("DrawObjects", tracy::Color::Blue);
	
	glm::mat4 view = mainLight_.get_view();

	glm::mat4 projection = mainLight_.get_projection();

	GPUCameraData camData;
	camData.proj = projection;
	camData.view = view;
	camData.viewproj = projection * view;
	
	//push data to dynmem
	uint32_t camera_data_offset = get_current_frame().dynamicData_.push(camData);
	 

	VkDescriptorBufferInfo objectBufferInfo = renderScene_.objectDataBuffer_.get_info();

	VkDescriptorBufferInfo camInfo = get_current_frame().dynamicData_.source_.get_info();
	camInfo.range = sizeof(GPUCameraData);

	VkDescriptorBufferInfo instanceInfo = pass.compactedInstanceBuffer_.get_info();


	VkDescriptorSet GlobalSet;
	vkutil::DescriptorBuilder::begin(descriptorLayoutCache_, get_current_frame().dynamicDescriptorAllocator_)
		.bind_buffer(0, &camInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, VK_SHADER_STAGE_VERTEX_BIT)
		.build(GlobalSet);

	VkDescriptorSet ObjectDataSet;
	vkutil::DescriptorBuilder::begin(descriptorLayoutCache_, get_current_frame().dynamicDescriptorAllocator_)
		.bind_buffer(0, &objectBufferInfo, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT)
		.bind_buffer(1, &instanceInfo, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT)
		.build(ObjectDataSet);

	vkCmdSetDepthBias(cmd, CVAR_ShadowBias.GetFloat(), 0, CVAR_SlopeBias.GetFloat());

	std::vector<uint32_t> dynamic_offsets;
	dynamic_offsets.push_back(camera_data_offset);

	execute_draw_commands(cmd, pass, ObjectDataSet, dynamic_offsets, GlobalSet);
}



struct alignas(16) DepthReduceData
{
	glm::vec2 imageSize;
};
inline uint32_t getGroupCount(uint32_t threadCount, uint32_t localSize)
{
	return (threadCount + localSize - 1) / localSize;
}

void VulkanEngine::reduce_depth(VkCommandBuffer cmd)
{
	vkutil::VulkanScopeTimer timer(cmd, profiler_, "Depth Reduce");

	VkImageMemoryBarrier depthReadBarriers[] =
	{
		vkinit::image_barrier(depthImage_.image_, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT),
	};

	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_DEPENDENCY_BY_REGION_BIT, 0, 0, 0, 0, 1, depthReadBarriers);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, depthReducePipeline_);

	for (int32_t i = 0; i < depthPyramidLevels_; ++i)
	{
		VkDescriptorImageInfo destTarget;
		destTarget.sampler = depthSampler_;
		destTarget.imageView = depthPyramidMips_[i];
		destTarget.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

		VkDescriptorImageInfo sourceTarget;
		sourceTarget.sampler = depthSampler_;
		if (i == 0)
		{
			sourceTarget.imageView = depthImage_.defaultView_;
			sourceTarget.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		}
		else {
			sourceTarget.imageView = depthPyramidMips_[i - 1];
			sourceTarget.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		}

		VkDescriptorSet depthSet;
		vkutil::DescriptorBuilder::begin(descriptorLayoutCache_, get_current_frame().dynamicDescriptorAllocator_)
			.bind_image(0, &destTarget, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT)
			.bind_image(1, &sourceTarget, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT)
			.build(depthSet);

		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, depthReduceLayout_, 0, 1, &depthSet, 0, nullptr);

		uint32_t levelWidth = depthPyramidWidth_ >> i;
		uint32_t levelHeight = depthPyramidHeight_ >> i;
		if (levelHeight < 1) levelHeight = 1;
		if (levelWidth < 1) levelWidth = 1;

		DepthReduceData reduceData = { glm::vec2(levelWidth, levelHeight) };

		vkCmdPushConstants(cmd, depthReduceLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(reduceData), &reduceData);
		vkCmdDispatch(cmd, getGroupCount(levelWidth, 32), getGroupCount(levelHeight, 32), 1);


		VkImageMemoryBarrier reduceBarrier = vkinit::image_barrier(depthPyramid_.image_, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT);

		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_DEPENDENCY_BY_REGION_BIT, 0, 0, 0, 0, 1, &reduceBarrier);
	}

	VkImageMemoryBarrier depthWriteBarrier = vkinit::image_barrier(depthImage_.image_, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);

	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, VK_DEPENDENCY_BY_REGION_BIT, 0, 0, 0, 0, 1, &depthWriteBarrier);

}