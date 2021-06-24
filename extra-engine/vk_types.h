// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

struct AllocatedBufferUntyped {
	VkBuffer buffer_{};
	VmaAllocation allocation_{};
	VkDeviceSize size_{0};
	VkDescriptorBufferInfo get_info(VkDeviceSize offset = 0);
};

template<typename T>
struct AllocatedBuffer : public AllocatedBufferUntyped {
	void operator=(const AllocatedBufferUntyped& other) {
		buffer_ = other.buffer_;
		allocation_ = other.allocation_;
		size_ = other.size_;
	}
	AllocatedBuffer(AllocatedBufferUntyped& other) {
		buffer_ = other.buffer_;
		allocation_ = other.allocation_;
		size_ = other.size_;
	}
	AllocatedBuffer() = default;
};

struct AllocatedImage {
	VkImage image_;
	VmaAllocation allocation_;
	VkImageView defaultView_;
	int mipLevels_;
};


inline VkDescriptorBufferInfo AllocatedBufferUntyped::get_info(VkDeviceSize offset)
{
	VkDescriptorBufferInfo info;
	info.buffer = buffer_;
	info.offset = offset;
	info.range = size_;
	return info;
}


enum class MeshpassType : uint8_t {
	None = 0,
	Forward = 1,
	Transparency = 2,
	DirectionalShadow = 3
};

