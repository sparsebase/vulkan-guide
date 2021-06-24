// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vk_types.h>
#include <vk_scene.h>

#include <vk_mesh.h>

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include <array>
#include <vector>
#include <unordered_map>
#include "material_system.h"

template<typename T>
struct Handle {
	uint32_t handle;
};

struct MeshObject;
struct Mesh;
struct GPUObjectData;
namespace vkutil { struct Material; }
namespace vkutil { struct ShaderPass; }

struct GPUIndirectObject {
	VkDrawIndexedIndirectCommand command;
	uint32_t objectID;
	uint32_t batchID;
};

struct DrawMesh {
	uint32_t firstVertex;
	uint32_t firstIndex;
	uint32_t indexCount;
	uint32_t vertexCount;
	bool isMerged;

	Mesh* original;
};



struct RenderObject {

	Handle<DrawMesh> meshID;
	Handle<vkutil::Material> material;

	uint32_t updateIndex;
	uint32_t customSortKey{0};

	vkutil::PerPassData<int32_t> passIndices;

	glm::mat4 transformMatrix;

	RenderBounds bounds;
};

struct GPUInstance {
	uint32_t objectID;
	uint32_t batchID;
};


class RenderScene {
public:
	struct PassMaterial {
		VkDescriptorSet materialSet;
		vkutil::ShaderPass* shaderPass;

		bool operator==(const PassMaterial& other) const
		{
			return materialSet == other.materialSet && shaderPass == other.shaderPass;
		}
	};
	struct PassObject {
		PassMaterial material;
		Handle<DrawMesh> meshID;
		Handle<RenderObject> original;
		int32_t builtbatch;
		uint32_t customKey;
	};
	struct RenderBatch {
		Handle<PassObject> object;
		uint64_t sortKey;

		bool operator==(const RenderBatch& other) const
		{
			return object.handle == other.object.handle && sortKey == other.sortKey;
		}
	};
	struct IndirectBatch {
		Handle<DrawMesh> meshID;
		PassMaterial material;
		uint32_t first;
		uint32_t count;
	};
	
	struct Multibatch {
		uint32_t first;
		uint32_t count;
	};
	struct MeshPass {

		std::vector<RenderScene::Multibatch> multibatches_;

		std::vector<RenderScene::IndirectBatch> batches_;

		std::vector<Handle<RenderObject>> unbatchedObjects_;

		std::vector<RenderScene::RenderBatch> flat_batches_;

		std::vector<PassObject> objects_;

		std::vector<Handle<PassObject>> reusableObjects_;

		std::vector<Handle<PassObject>> objectsToDelete_;

		
		AllocatedBuffer<uint32_t> compactedInstanceBuffer_;
		AllocatedBuffer<GPUInstance> passObjectsBuffer_;

		AllocatedBuffer<GPUIndirectObject> drawIndirectBuffer_;
		AllocatedBuffer<GPUIndirectObject> clearIndirectBuffer_;

		PassObject* get(Handle<PassObject> handle);

		MeshpassType type_;

		bool needsIndirectRefresh_ = true;
		bool needsInstanceRefresh_ = true;
	};

	void init();

	Handle<RenderObject> register_object(MeshObject* object);

	void register_object_batch(MeshObject* first, uint32_t count);

	void update_transform(Handle<RenderObject> objectID,const glm::mat4 &localToWorld);
	void update_object(Handle<RenderObject> objectID);
	
	void fill_objectData(GPUObjectData* data);
	void fill_indirectArray(GPUIndirectObject* data, MeshPass& pass);
	void fill_instancesArray(GPUInstance* data, MeshPass& pass);

	void write_object(GPUObjectData* target, Handle<RenderObject> objectID);
	
	void clear_dirty_objects();

	void build_batches();

	void merge_meshes(class VulkanEngine* engine);

	void refresh_pass(MeshPass* pass);

	void build_indirect_batches(MeshPass* pass, std::vector<IndirectBatch>& outbatches, std::vector<RenderScene::RenderBatch>& inobjects);
	RenderObject* get_object(Handle<RenderObject> objectID);
	DrawMesh* get_mesh(Handle<DrawMesh> objectID);

	vkutil::Material *get_material(Handle<vkutil::Material> objectID);

	std::vector<RenderObject> renderables_;
	std::vector<DrawMesh> meshes_;
	std::vector<vkutil::Material*> materials_;

	std::vector<Handle<RenderObject>> dirtyObjects_;

	MeshPass* get_mesh_pass(MeshpassType name);

	MeshPass forwardPass_;
	MeshPass transparentForwardPass_;
	MeshPass shadowPass_;

	std::unordered_map<vkutil::Material*, Handle<vkutil::Material>> materialConvert_;
	std::unordered_map<Mesh*, Handle<DrawMesh>> meshConvert_;

	Handle<vkutil::Material> getMaterialHandle(vkutil::Material* m);
	Handle<DrawMesh> getMeshHandle(Mesh* m);
	

	AllocatedBuffer<Vertex> mergedVertexBuffer_;
	AllocatedBuffer<uint32_t> mergedIndexBuffer_;

	AllocatedBuffer<GPUObjectData> objectDataBuffer_;
};

