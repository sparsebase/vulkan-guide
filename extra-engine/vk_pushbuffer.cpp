#include <vk_pushbuffer.h>

uint32_t vkutil::PushBuffer::push(void* data, size_t size)
{
	uint32_t offset = currentOffset_;
	char* target = (char*)mapped_;
	target += currentOffset_;
	memcpy(target, data, size);
	currentOffset_ += static_cast<uint32_t>(size);
	currentOffset_ = pad_uniform_buffer_size(currentOffset_);

	return offset;
}

void vkutil::PushBuffer::init(VmaAllocator& allocator, AllocatedBufferUntyped sourceBuffer, uint32_t alignement)
{
	align_ = alignement;
	source_ = sourceBuffer;
	currentOffset_ = 0;
	vmaMapMemory(allocator, sourceBuffer.allocation_, &mapped_);
}

void vkutil::PushBuffer::reset()
{
	currentOffset_ = 0;
}

uint32_t vkutil::PushBuffer::pad_uniform_buffer_size(uint32_t originalSize)
{
	// Calculate required alignment based on minimum device offset alignment
	size_t minUboAlignment = align_;
	size_t alignedSize = originalSize;
	if (minUboAlignment > 0) {
		alignedSize = (alignedSize + minUboAlignment - 1) & ~(minUboAlignment - 1);
	}
	return static_cast<uint32_t>(alignedSize);
}