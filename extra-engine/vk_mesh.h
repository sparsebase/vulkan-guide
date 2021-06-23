#pragma once

#include <vk_types.h>
#include <vector>
#include <glm/vec3.hpp>
#include <glm/vec2.hpp>

constexpr bool logMeshUpload = false;


struct VertexInputDescription {
	std::vector<VkVertexInputBindingDescription> bindings;
	std::vector<VkVertexInputAttributeDescription> attributes;

	VkPipelineVertexInputStateCreateFlags flags = 0;
};



struct Vertex {

	glm::vec3 position;
	//glm::vec3 normal;
	glm::vec<2, uint8_t> oct_normal;//color;
	glm::vec<3, uint8_t> color;
	glm::vec2 uv;
	static VertexInputDescription get_vertex_description();

	void pack_normal(glm::vec3 n);
	void pack_color(glm::vec3 c);
};
struct RenderBounds {
	glm::vec3 origin;
	float radius;
	glm::vec3 extents;
	bool valid;
};
struct Mesh {
	std::vector<Vertex> _vertices;
	std::vector<uint32_t> _indices;

	AllocatedBuffer<Vertex> _vertexBuffer;
	AllocatedBuffer<uint32_t> _indexBuffer;

	RenderBounds bounds;

	bool load_from_meshasset(const char* filename);
};