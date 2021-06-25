// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vk_types.h>
#include <vector>
#include <functional>
#include <deque>
#include <memory>
#include <vk_mesh.h>
#include <vk_scene.h>
#include <vk_shaders.h>
#include <vk_pushbuffer.h>
#include <player_camera.h>
#include <unordered_map>
#include <material_system.h>



#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include <SDL_events.h>
#include <frustum_cull.h>

namespace vkutil { struct Material; }


namespace tracy { class VkCtx; }


namespace assets { struct PrefabInfo; }


//forward declarations
namespace vkutil {
	class DescriptorLayoutCache;
	class DescriptorAllocator;
	class VulkanProfiler;
}

struct DeletionQueue
{
    std::deque<std::function<void()>> deletors;

	template<typename F>
    void push_function(F&& function) {
		static_assert(sizeof(F) < 200, "DONT CAPTURE TOO MUCH IN THE LAMBDA");
        deletors.push_back(function);
    }

    void flush() {
        // reverse iterate the deletion queue to execute all the functions
        for (auto it = deletors.rbegin(); it != deletors.rend(); it++) {
            (*it)(); //call functors
        }

        deletors.clear();
    }
};

struct MeshPushConstants {
	glm::vec4 data;
	glm::mat4 render_matrix;
};

namespace assets {

	enum class TransparencyMode :uint8_t;
}

struct Texture {
	AllocatedImage image;
	VkImageView imageView;
};



struct MeshObject {
	Mesh* mesh{ nullptr };

	vkutil::Material* material;
	uint32_t customSortKey;
	glm::mat4 transformMatrix;

	RenderBounds bounds;

	uint32_t bDrawForwardPass : 1;
	uint32_t bDrawShadowPass : 1;
};


struct FrameData {
	VkSemaphore presentSemaphore_, renderSemaphore_;
	VkFence renderFence_;

	DeletionQueue frameDeletionQueue_;

	VkCommandPool commandPool_;
	VkCommandBuffer mainCommandBuffer_;
	
	vkutil::PushBuffer dynamicData_;
	//AllocatedBufferUntyped dynamicDataBuffer;

	AllocatedBufferUntyped debugOutputBuffer_;

	vkutil::DescriptorAllocator* dynamicDescriptorAllocator_;

	std::vector<uint32_t> debugDataOffsets_;
	std::vector<std::string> debugDataNames_;
};


struct UploadContext {
	VkFence uploadFence_;
	VkCommandPool commandPool_;	
};

struct GPUCameraData{
	glm::mat4 view;
	glm::mat4 proj;
	glm::mat4 viewproj;
};


struct GPUSceneData {
	glm::vec4 fogColor; // w is for exponent
	glm::vec4 fogDistances; //x for min, y for max, zw unused.
	glm::vec4 ambientColor;
	glm::vec4 sunlightDirection; //w for sun power
	glm::vec4 sunlightColor;
	glm::mat4 sunlightShadowMatrix;
};

struct GPUObjectData {
	glm::mat4 modelMatrix;
	glm::vec4 origin_rad; // bounds
	glm::vec4 extents;  // bounds
};


struct EngineStats {
	float frametime;
	int objects;
	int drawcalls;
	int draws;
	int triangles;
};


struct MeshDrawCommands {
	struct RenderBatch {
		MeshObject* object;
		uint64_t sortKey;
		uint64_t objectIndex;
	};

	std::vector<RenderBatch> batch;
};

struct /*alignas(16)*/DrawCullData
{
	glm::mat4 viewMat;
	float P00, P11, znear, zfar; // symmetric projection parameters
	float frustum[4]; // data for left/right/top/bottom frustum planes
	float lodBase, lodStep; // lod distance i = base * pow(step, i)
	float pyramidWidth, pyramidHeight; // depth pyramid size in texels

	uint32_t drawCount;

	int cullingEnabled;
	int lodEnabled;
	int occlusionEnabled;
	int distanceCheck;
	int AABBcheck;
	float aabbmin_x;
	float aabbmin_y;
	float aabbmin_z;
	float aabbmax_x;
	float aabbmax_y;
	float aabbmax_z;
	
};

//struct EngineConfig {
//	//float drawDistance{5000};
//	//float shadowBias{ 5.25f };
//	//float shadowBiasslope{4.75f };
//	//bool occlusionCullGPU{ true };
//	//bool frustrumCullCPU{ true };
//	//bool outputIndirectBufferToFile{false};
//	//bool freezeCulling{ false };
//	//bool mouseLook{ true };
//};

struct DirectionalLight {
	glm::vec3 lightPosition;
	glm::vec3 lightDirection;
	glm::vec3 shadowExtent;
	glm::mat4 get_projection();

	glm::mat4 get_view();
};
struct CullParams {
	glm::mat4 viewmat;
	glm::mat4 projmat;
	bool occlusionCull;
	bool frustrumCull;
	float drawDist;
	bool aabb;
	glm::vec3 aabbmin;
	glm::vec3 aabbmax;
};
constexpr unsigned int FRAME_OVERLAP = 2;
const int MAX_OBJECTS = 150000;
class VulkanEngine {
public:

	bool isInitialized_{ false };
	int frameNumber_ {0};
	int selectedShader_{ 0 };

	VkExtent2D windowExtent_{ 1700 , 900 };

	struct SDL_Window* window_{ nullptr };

	VkInstance instance_;
	VkPhysicalDevice physicalDevice_;
	VkDevice device_;

	VkPhysicalDeviceProperties gpuProperties_;

	FrameData frames_[FRAME_OVERLAP];
	
	VkQueue graphicsQueue_;
	uint32_t graphicsQueueFamily_;
	
	tracy::VkCtx* graphicsQueueContext_;

	VkRenderPass renderPass_;
	VkRenderPass shadowPass_;
	VkRenderPass copyPass_;

	VkSurfaceKHR surface_;
	VkSwapchainKHR swapchain_;
	VkFormat swachainImageFormat_;

	VkFormat renderFormat_;
	AllocatedImage rawRenderImage_;
	VkSampler smoothSampler_;
	VkFramebuffer forwardFramebuffer_;
	VkFramebuffer shadowFramebuffer_;
	std::vector<VkFramebuffer> framebuffers_;
	std::vector<VkImage> swapchainImages_;
	std::vector<VkImageView> swapchainImageViews_;	

    DeletionQueue mainDeletionQueue_;
	
	VmaAllocator allocator_; //vma lib allocator

	//depth resources
	
	AllocatedImage depthImage_;
	AllocatedImage depthPyramid_;
	VkSampler shadowSampler_;
	AllocatedImage shadowImage_;
	//VkExtent2D _shadowExtent{1024,1024};
	VkExtent2D shadowExtent_{ 1024*4,1024*4 };
	int depthPyramidWidth_ ;
	int depthPyramidHeight_;
	int depthPyramidLevels_;
	
	//the format for the depth image
	VkFormat depthFormat_;
	
	vkutil::DescriptorAllocator* descriptorAllocator_;
	vkutil::DescriptorLayoutCache* descriptorLayoutCache_;
	vkutil::VulkanProfiler* profiler_;
	vkutil::MaterialSystem* materialSystem_;

	//VkDescriptorSetLayout singleTextureSetLayout_;

	GPUSceneData sceneParameters_;

	std::vector<VkBufferMemoryBarrier> uploadBarriers_;

	std::vector<VkBufferMemoryBarrier> cullReadyBarriers_;

	std::vector<VkBufferMemoryBarrier> postCullBarriers_;

	UploadContext uploadContext_;

	PlayerCamera camera_;
	DirectionalLight mainLight_;

	VkPipeline cullPipeline_;
	VkPipelineLayout cullLayout_;

	VkPipeline depthReducePipeline_;
	VkPipelineLayout depthReduceLayout_;

	VkPipeline sparseUploadPipeline_;
	VkPipelineLayout sparseUploadLayout_;

	VkPipeline blitPipeline_;
	VkPipelineLayout blitLayout_;

	VkSampler depthSampler_;
	VkImageView depthPyramidMips_[16] = {};

	MeshDrawCommands currentCommands_;
	RenderScene renderScene_;

	//EngineConfig _config;

	void ready_mesh_draw(VkCommandBuffer cmd);
	
	//initializes everything in the engine
	void init();

	//shuts down the engine
	void cleanup();

	//draw loop
	void draw();

	void forward_pass(VkClearValue clearValue, VkCommandBuffer cmd);

	void shadow_pass(VkCommandBuffer cmd);
	

	//run main loop
	void run();
	
	FrameData& get_current_frame();
	FrameData& get_last_frame();

	ShaderCache shaderCache_;

	std::unordered_map<std::string, Mesh> meshes_;
	std::unordered_map<std::string, Texture> loadedTextures_;
	std::unordered_map<std::string, assets::PrefabInfo*> prefabCache_;
	//functions

	//returns nullptr if it cant be found
	Mesh* get_mesh(const std::string& name);

	//our draw function
	void draw_objects_forward(VkCommandBuffer cmd, RenderScene::MeshPass& pass);

	void execute_draw_commands(VkCommandBuffer cmd, RenderScene::MeshPass& pass, VkDescriptorSet ObjectDataSet, std::vector<uint32_t> dynamic_offsets, VkDescriptorSet GlobalSet);

	void draw_objects_shadow(VkCommandBuffer cmd, RenderScene::MeshPass& pass);
	
	void reduce_depth(VkCommandBuffer cmd);

	void execute_compute_cull(VkCommandBuffer cmd, RenderScene::MeshPass& pass,CullParams& params);

	void ready_cull_data(RenderScene::MeshPass& pass, VkCommandBuffer cmd);

	AllocatedBufferUntyped create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage, VkMemoryPropertyFlags required_flags = 0);

	void reallocate_buffer(AllocatedBufferUntyped&buffer,size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage, VkMemoryPropertyFlags required_flags = 0);


	size_t pad_uniform_buffer_size(size_t originalSize);

	void immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function);

	bool load_prefab(const char* path, glm::mat4 root);

	static std::string asset_path(std::string_view path);
	
	static std::string shader_path(std::string_view path);
	void refresh_renderbounds(MeshObject* object);

	template<typename T>
	T* map_buffer(AllocatedBuffer<T> &buffer);
	
	void unmap_buffer(AllocatedBufferUntyped& buffer);

	bool load_compute_shader(const char* shaderPath, VkPipeline& pipeline, VkPipelineLayout& layout);
private:
	EngineStats stats;
	void process_input_event(SDL_Event* ev);

	void init_vulkan();

	void init_swapchain();

	void init_forward_renderpass();

	void init_copy_renderpass();

	void init_shadow_renderpass();

	void init_framebuffers();

	void init_commands();

	void init_sync_structures();

	void init_pipelines();

	void init_scene();

	void init_descriptors();

	void init_imgui();	

	void load_meshes();

	void load_images();

	bool load_image_to_cache(const char* name, const char* path);

	void upload_mesh(Mesh& mesh);

	void copy_render_to_swapchain(uint32_t swapchainImageIndex, VkCommandBuffer cmd);
};

template<typename T>
T* VulkanEngine::map_buffer(AllocatedBuffer<T>& buffer)
{
	void* data;
	vmaMapMemory(allocator_, buffer.allocation_, &data);
	return(T*)data;
}
