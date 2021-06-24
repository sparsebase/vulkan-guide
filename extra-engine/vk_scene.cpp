#include <vk_scene.h>
#include <vk_engine.h>
#include "Tracy.hpp"
#include "logger.h"

void RenderScene::init()
{
	forwardPass_.type_ = MeshpassType::Forward;
	shadowPass_.type_ = MeshpassType::DirectionalShadow;
	transparentForwardPass_.type_ = MeshpassType::Transparency;
}

Handle<RenderObject> RenderScene::register_object(MeshObject* object)
{
	RenderObject newObj;
	newObj.bounds = object->bounds;
	newObj.transformMatrix = object->transformMatrix;	
	newObj.material = getMaterialHandle(object->material);
	newObj.meshID = getMeshHandle(object->mesh);
	newObj.updateIndex = (uint32_t)-1;
	newObj.customSortKey = object->customSortKey;
	newObj.passIndices.clear(-1);
	Handle<RenderObject> handle;
	handle.handle = static_cast<uint32_t>(renderables_.size());
	
	renderables_.push_back(newObj);

	if (object->bDrawForwardPass)
	{
		if (object->material->original->passShaders[MeshpassType::Transparency])
		{
			transparentForwardPass_.unbatchedObjects_.push_back(handle);
		}
		if (object->material->original->passShaders[MeshpassType::Forward])
		{
			forwardPass_.unbatchedObjects_.push_back(handle);
		}
	}
	if (object->bDrawShadowPass)
	{
		if (object->material->original->passShaders[MeshpassType::DirectionalShadow])
		{
			shadowPass_.unbatchedObjects_.push_back(handle);
		}
	}

	update_object(handle);
	return handle;
}

void RenderScene::register_object_batch(MeshObject* first, uint32_t count)
{
	renderables_.reserve(count);

	for (uint32_t i = 0; i < count; i++) {
		register_object(&(first[i]));
	}
}

void RenderScene::update_transform(Handle<RenderObject> objectID, const glm::mat4& localToWorld)
{
	get_object(objectID)->transformMatrix = localToWorld;
	update_object(objectID);
}


void RenderScene::update_object(Handle<RenderObject> objectID)
{
	auto& passIndices = get_object(objectID)->passIndices;
	if (passIndices[MeshpassType::Forward] != -1)
	{
		Handle<PassObject> obj;
		obj.handle = passIndices[MeshpassType::Forward];

		forwardPass_.objectsToDelete_.push_back(obj);
		forwardPass_.unbatchedObjects_.push_back(objectID);

		passIndices[MeshpassType::Forward] = -1;
	}


	if (passIndices[MeshpassType::DirectionalShadow] != -1)
	{
		Handle<PassObject> obj;
		obj.handle = passIndices[MeshpassType::DirectionalShadow];

		shadowPass_.objectsToDelete_.push_back(obj);
		shadowPass_.unbatchedObjects_.push_back(objectID);

		passIndices[MeshpassType::DirectionalShadow] = -1;
	}


	if (passIndices[MeshpassType::Transparency] != -1)
	{
		Handle<PassObject> obj;
		obj.handle = passIndices[MeshpassType::Transparency];

		transparentForwardPass_.unbatchedObjects_.push_back(objectID);
		transparentForwardPass_.objectsToDelete_.push_back(obj);

		passIndices[MeshpassType::Transparency] = -1;
	}

	
	if (get_object(objectID)->updateIndex == (uint32_t)-1)
	{

		get_object(objectID)->updateIndex = static_cast<uint32_t>(dirtyObjects_.size());

		dirtyObjects_.push_back(objectID);
	}
}

void RenderScene::write_object(GPUObjectData* target, Handle<RenderObject> objectID)
{
	RenderObject* renderable = get_object(objectID);
	GPUObjectData object;

	object.modelMatrix = renderable->transformMatrix;
	object.origin_rad = glm::vec4(renderable->bounds.origin, renderable->bounds.radius);
	object.extents = glm::vec4(renderable->bounds.extents, renderable->bounds.valid ? 1.f : 0.f);

	memcpy(target, &object, sizeof(GPUObjectData));
}

void RenderScene::fill_objectData(GPUObjectData* data)
{
	
	for(int i = 0; i < renderables_.size(); i++)
	{
		Handle<RenderObject> h;
		h.handle = i;
		write_object(data + i, h);
	}
}


void RenderScene::fill_indirectArray(GPUIndirectObject* data, MeshPass& pass)
{
	ZoneScopedNC("Fill Indirect", tracy::Color::Red);
	int dataIndex = 0;
	for (int i = 0; i < pass.batches_.size(); i++) {

		auto batch = pass.batches_[i];

		data[dataIndex].command.firstInstance = batch.first;//i;
		data[dataIndex].command.instanceCount = 0;
		data[dataIndex].command.firstIndex = get_mesh(batch.meshID)->firstIndex;
		data[dataIndex].command.vertexOffset = get_mesh(batch.meshID)->firstVertex;
		data[dataIndex].command.indexCount = get_mesh(batch.meshID)->indexCount;
		data[dataIndex].objectID = 0;
		data[dataIndex].batchID = i;

		dataIndex++;
	}
}

void RenderScene::fill_instancesArray(GPUInstance* data, MeshPass& pass)
{
	ZoneScopedNC("Fill Instances", tracy::Color::Red);
	int dataIndex = 0;
	for (int i = 0; i < pass.batches_.size(); i++) {

		auto batch = pass.batches_[i];

		
		for (int b = 0; b < batch.count; b++)
		{
			
			data[dataIndex].objectID = pass.get(pass.flat_batches_[b + batch.first].object)->original.handle;
			data[dataIndex].batchID = i;
			dataIndex++;
		}
	}
}

void RenderScene::clear_dirty_objects()
{
	for (auto obj : dirtyObjects_)
	{
		get_object(obj)->updateIndex = (uint32_t)-1;
	}
	dirtyObjects_.clear();
}
#include <future>
void RenderScene::build_batches()
{
#if 1
	auto fwd = std::async(std::launch::async, [&] { refresh_pass(&forwardPass_); });
	auto shadow = std::async(std::launch::async, [&] { refresh_pass(&shadowPass_); });
	auto transparent = std::async(std::launch::async, [&] { refresh_pass(&transparentForwardPass_); });

	transparent.get();
	shadow.get();
	fwd.get();
#else
	refresh_pass(&forwardPass_);
	refresh_pass(&transparentForwardPass_);
	refresh_pass(&shadowPass_);
#endif
	
}

void RenderScene::merge_meshes(VulkanEngine* engine)
{
	ZoneScopedNC("Mesh Merge", tracy::Color::Magenta)
	size_t total_vertices = 0;
	size_t total_indices = 0;

	for (auto& m : meshes_)
	{
		m.firstIndex = static_cast<uint32_t>(total_indices);
		m.firstVertex = static_cast<uint32_t>(total_vertices);

		total_vertices += m.vertexCount;
		total_indices += m.indexCount;

		m.isMerged = true;
	}

	mergedVertexBuffer_ = engine->create_buffer(total_vertices * sizeof(Vertex), VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, 
		VMA_MEMORY_USAGE_GPU_ONLY);

	mergedIndexBuffer_ = engine->create_buffer(total_indices * sizeof(uint32_t), VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
		VMA_MEMORY_USAGE_GPU_ONLY);

	engine->immediate_submit([&](VkCommandBuffer cmd)
	{
		for (auto& m : meshes_)
		{
			VkBufferCopy vertexCopy;
			vertexCopy.dstOffset = m.firstVertex * sizeof(Vertex);
			vertexCopy.size = m.vertexCount * sizeof(Vertex);
			vertexCopy.srcOffset = 0;

			vkCmdCopyBuffer(cmd, m.original->vertexBuffer_.buffer_, mergedVertexBuffer_.buffer_, 1, &vertexCopy);

			VkBufferCopy indexCopy;
			indexCopy.dstOffset = m.firstIndex * sizeof(uint32_t);
			indexCopy.size = m.indexCount * sizeof(uint32_t);
			indexCopy.srcOffset = 0;

			vkCmdCopyBuffer(cmd, m.original->indexBuffer_.buffer_, mergedIndexBuffer_.buffer_, 1, &indexCopy);
		}
	});
}

void RenderScene::refresh_pass(MeshPass* pass)
{
	pass->needsIndirectRefresh_ = true;
	pass->needsInstanceRefresh_ = true;

	std::vector<uint32_t> new_objects;
	if(pass->objectsToDelete_.size() > 0)
	{
		ZoneScopedNC("Delete objects", tracy::Color::Blue3);

		//create the render batches so that then we can do the deletion on the flat-array directly

		std::vector<RenderScene::RenderBatch> deletion_batches;
		deletion_batches.reserve(new_objects.size());
		
	
		for (auto i : pass->objectsToDelete_) {
			pass->reusableObjects_.push_back(i);
			RenderScene::RenderBatch newCommand;

			auto obj = pass->objects_[i.handle];
			newCommand.object= i;

			uint64_t pipelinehash = std::hash<uint64_t>()(uint64_t(obj.material.shaderPass->pipeline));
			uint64_t sethash = std::hash<uint64_t>()((uint64_t)obj.material.materialSet);

			uint32_t mathash = static_cast<uint32_t>(pipelinehash ^ sethash);

			uint32_t meshmat = uint64_t(mathash) ^ uint64_t(obj.meshID.handle);

			//pack mesh id and material into 64 bits				
			newCommand.sortKey = uint64_t(meshmat) | (uint64_t(obj.customKey) << 32);

			pass->objects_[i.handle].customKey = 0;
			pass->objects_[i.handle].material.shaderPass = nullptr;
			pass->objects_[i.handle].meshID.handle = -1;
			pass->objects_[i.handle].original.handle = -1;

			deletion_batches.push_back(newCommand);
			
		}
		pass->objectsToDelete_.clear();
		{
			ZoneScopedNC("Deletion Sort", tracy::Color::Blue1);
			std::sort(deletion_batches.begin(), deletion_batches.end(), [](const RenderScene::RenderBatch& A, const RenderScene::RenderBatch& B) {
				if (A.sortKey < B.sortKey) { return true; }
				else if (A.sortKey == B.sortKey) { return A.object.handle < B.object.handle; }
				else { return false; }
			});
		}
		{
			ZoneScopedNC("removal", tracy::Color::Blue1);

			std::vector<RenderScene::RenderBatch> newbatches;
			newbatches.reserve(pass->flat_batches_.size());

			{
				ZoneScopedNC("Set Difference", tracy::Color::Red);

				std::set_difference(pass->flat_batches_.begin(), pass->flat_batches_.end(), deletion_batches.begin(), deletion_batches.end(), std::back_inserter(newbatches), [](const RenderScene::RenderBatch& A, const RenderScene::RenderBatch& B) {
					if (A.sortKey < B.sortKey) { return true; }
					else if (A.sortKey == B.sortKey) { return A.object.handle < B.object.handle; }
					else { return false; }
				});
			}
			pass->flat_batches_ = std::move(newbatches);
		}
	}
	{
		ZoneScopedNC("Fill ObjectList", tracy::Color::Blue2);
			
		new_objects.reserve(pass->unbatchedObjects_.size());
		for (auto o : pass->unbatchedObjects_)
		{
			RenderScene::PassObject newObject;

			newObject.original = o;
			newObject.meshID = get_object(o)->meshID;

			//pack mesh id and material into 32 bits
			vkutil::Material* mt = get_material(get_object(o)->material);
			newObject.material.materialSet = mt->passSets[pass->type_];
			newObject.material.shaderPass = mt->original->passShaders[pass->type_];
			newObject.customKey = get_object(o)->customSortKey;

			uint32_t handle = -1;

			//reuse handle
			if (pass->reusableObjects_.size() > 0)
			{
				handle = pass->reusableObjects_.back().handle;
				pass->reusableObjects_.pop_back();
				pass->objects_[handle] = newObject;
			}
			else 
			{
				handle = pass->objects_.size();
				pass->objects_.push_back(newObject);
			}

			
			new_objects.push_back(handle);
			get_object(o)->passIndices[pass->type_] = static_cast<int32_t>(handle);
		}

		pass->unbatchedObjects_.clear();
	}

	std::vector<RenderScene::RenderBatch> new_batches;
	new_batches.reserve(new_objects.size());

	{
		ZoneScopedNC("Fill DrawList", tracy::Color::Blue2);	
		
		for (auto i : new_objects) {
			{
				RenderScene::RenderBatch newCommand;

				auto obj = pass->objects_[i];
				newCommand.object.handle = i;

				uint64_t pipelinehash = std::hash<uint64_t>()(uint64_t(obj.material.shaderPass->pipeline));
				uint64_t sethash = std::hash<uint64_t>()((uint64_t)obj.material.materialSet);

				uint32_t mathash = static_cast<uint32_t>(pipelinehash ^ sethash);
				
				uint32_t meshmat = uint64_t(mathash) ^ uint64_t(obj.meshID.handle);

				//pack mesh id and material into 64 bits				
				newCommand.sortKey = uint64_t(meshmat) | (uint64_t(obj.customKey) << 32);

				new_batches.push_back(newCommand);
			}
		}
	}

	{
		ZoneScopedNC("Draw Sort", tracy::Color::Blue1);
		std::sort(new_batches.begin(), new_batches.end(), [](const RenderScene::RenderBatch& A, const RenderScene::RenderBatch& B) {
			if (A.sortKey < B.sortKey) { return true; }
			else if (A.sortKey == B.sortKey) { return A.object.handle < B.object.handle; }
			else { return false; }
		});
	}
	{
		ZoneScopedNC("Draw Merge batches", tracy::Color::Blue2);

		//merge the new batches into the main batch array

		if (pass->flat_batches_.size() > 0 && new_batches.size() > 0)
		{
			size_t index = pass->flat_batches_.size();
			pass->flat_batches_.reserve(pass->flat_batches_.size() + new_batches.size());
			
			for (auto b : new_batches)
			{
				pass->flat_batches_.push_back(b);
			}

			RenderScene::RenderBatch* begin = pass->flat_batches_.data();
			RenderScene::RenderBatch* mid = begin + index;
			RenderScene::RenderBatch* end = begin + pass->flat_batches_.size();
			//std::sort(pass->flat_batches.begin(), pass->flat_batches.end(), [](const RenderScene::RenderBatch& A, const RenderScene::RenderBatch& B) {
			//	return A.sortKey < B.sortKey;
			//	});
			std::inplace_merge(begin, mid, end, [](const RenderScene::RenderBatch& A, const RenderScene::RenderBatch& B) {
				if (A.sortKey < B.sortKey) { return true; }
				else if (A.sortKey == B.sortKey) { return A.object.handle < B.object.handle; }
				else { return false; }
			});
		}
		else if (pass->flat_batches_.size() == 0)
		{
			pass->flat_batches_ = std::move(new_batches);
		}
	}
	
	{
		ZoneScopedNC("Draw Merge", tracy::Color::Blue);

		pass->batches_.clear();

		build_indirect_batches(pass,pass->batches_,pass->flat_batches_);

		//flatten batches into multibatch
		Multibatch newbatch;
		pass->multibatches_.clear();

		
		newbatch.count = 1;
		newbatch.first = 0;
		 
#if 1
		for (int i = 1; i < pass->batches_.size(); i++)
		{
			IndirectBatch* joinbatch = &pass->batches_[newbatch.first];
			IndirectBatch* batch = &pass->batches_[i];

			
			bool bCompatibleMesh = get_mesh(joinbatch->meshID)->isMerged;
			
					
			bool bSameMat = false;
			
			if (bCompatibleMesh && joinbatch->material.materialSet == batch->material.materialSet &&
				joinbatch->material.shaderPass == batch->material.shaderPass
				)
			{
				bSameMat = true;
			}


			if (!bSameMat || !bCompatibleMesh)
			{
				pass->multibatches_.push_back(newbatch);
				newbatch.count = 1;
				newbatch.first = i;
			}
			else {
				newbatch.count++;
			}
		}
		pass->multibatches_.push_back(newbatch);
#else 
		for (int i = 0; i < pass->batches_.size(); i++)
		{
			Multibatch newbatch;
			newbatch.count = 1;
			newbatch.first = i;

			pass->multibatches_.push_back(newbatch);
		}
#endif
	}
}

void RenderScene::build_indirect_batches(MeshPass* pass, std::vector<IndirectBatch>& outbatches, std::vector<RenderScene::RenderBatch>& inobjects)
{
	if (inobjects.size() == 0) return;

	ZoneScopedNC("Build Indirect Batches", tracy::Color::Blue);

	RenderScene::IndirectBatch newBatch;
	newBatch.first = 0;
	newBatch.count = 0;

	newBatch.material = pass->get(inobjects[0].object)->material;
	newBatch.meshID = pass->get(inobjects[0].object)->meshID;

	outbatches.push_back(newBatch);
	RenderScene::IndirectBatch* back = &pass->batches_.back();

	RenderScene::PassMaterial lastMat = pass->get(inobjects[0].object)->material;
	for (int i = 0; i <inobjects.size(); i++) {
		PassObject* obj = pass->get(inobjects[i].object);

		bool bSameMesh = obj->meshID.handle == back->meshID.handle;
		bool bSameMaterial = false;
		if (obj->material == lastMat)
		{
			bSameMaterial = true;
		}

		if (!bSameMaterial || !bSameMesh)
		{
			newBatch.material = obj->material;

			if (newBatch.material == back->material)
			{
				bSameMaterial = true;
			}
		}

		if (bSameMesh && bSameMaterial)
		{
			back->count++;
		}
		else {

			newBatch.first = i;
			newBatch.count = 1;
			newBatch.meshID = obj->meshID;

			outbatches.push_back(newBatch);
			back = &outbatches.back();
		}
		//back->objects.push_back(obj->original);
	}
}

RenderObject* RenderScene::get_object(Handle<RenderObject> objectID)
{
	return &renderables_[objectID.handle];
}

DrawMesh* RenderScene::get_mesh(Handle<DrawMesh> objectID)
{
	return &meshes_[objectID.handle];
}

vkutil::Material* RenderScene::get_material(Handle<vkutil::Material> objectID)
{
	return materials_[objectID.handle];
}

RenderScene::MeshPass* RenderScene::get_mesh_pass(MeshpassType name)
{
	switch (name)
	{	
	case MeshpassType::Forward:
		return &forwardPass_;
		break;
	case MeshpassType::Transparency:
		return &transparentForwardPass_;
		break;
	case MeshpassType::DirectionalShadow:
		return &shadowPass_;
		break;
	}
	return nullptr;
}

Handle<vkutil::Material> RenderScene::getMaterialHandle(vkutil::Material* m)
{	
	Handle<vkutil::Material> handle;
	auto it = materialConvert_.find(m);
	if (it == materialConvert_.end())
	{
		uint32_t index = static_cast<uint32_t>(materials_.size());
		materials_.push_back(m);

		handle.handle = index;
		materialConvert_[m] = handle;
	}
	else {
		handle = (*it).second;
	}
	return handle;
}

Handle<DrawMesh> RenderScene::getMeshHandle(Mesh* m)
{
	Handle<DrawMesh> handle;
	auto it = meshConvert_.find(m);
	if (it == meshConvert_.end())
	{
		uint32_t index = static_cast<uint32_t>(meshes_.size());

		DrawMesh newMesh;
		newMesh.original = m;
		newMesh.firstIndex = 0;
		newMesh.firstVertex = 0;
		newMesh.vertexCount = static_cast<uint32_t>(m->vertices_.size());
		newMesh.indexCount = static_cast<uint32_t>(m->indices_.size());

		meshes_.push_back(newMesh);

		handle.handle = index;
		meshConvert_[m] = handle;
	}
	else {
		handle = (*it).second;
	}
	return handle;
}

RenderScene::PassObject* RenderScene::MeshPass::get(Handle<PassObject> handle)
{
	return &objects_[handle.handle];
}
