#pragma once

#include <c10/cuda/CUDAMacros.h>

#include <cuda_runtime_api.h>
#include <list>
#include <map>
#include <vector>
#include <cstdint>
#include <atomic>

namespace c10 {

namespace cuda::GPUMemAllocator {
/**
 * We implement a naive version of GPU memory allocator
 * Change all cudaMalloc/cudaFree(or some API like these) to new API
 */

/**
 * @brief A struct to store the information of a block(stored in the host
 * memory)
 * @param dev_ptr: the pointer to GPU side memory block start address
 * @param order: the order of the block in the free list, the block size is (1
 * << order) bytes
 * @param pred: the pointer to the previous block(to find its buddy)
 * @param succ: the pointer to the next block(to find its buddy)
 */
struct block_info {
  void* dev_ptr;
  bool is_free;
  std::uint32_t order;
  block_info* pred;
  block_info* succ;
};

/**
 * @brief A class to manage GPU memory allocation(using Buddy System)
 * @param free_lists: a vector of free lists, each list contains blocks of the
 * same size
 * @param block_map: a map to store the mapping from a block's start address to
 * its block_info
 * @param device_ptr: the pointer to the start address of the GPU memory
 * @param is_unified: whether to use unified memory allocation
 * @param TOTAL_MEM: the total size of the GPU memory
 * @param MAX_ORDER: the maximum order of the blocks in the free lists
 */

class GPUMemAllocator {
 private:
  std::vector<std::list<block_info*>> free_lists;
  std::map<void*, block_info*> block_map;
  void* device_ptr;
  bool is_unified;
  std::size_t TOTAL_MEM;
  std::size_t MAX_ORDER;
  bool is_robust;

 public:
  GPUMemAllocator(std::size_t total_mem = (1 << 30), bool is_robust = false);
  ~GPUMemAllocator();

  /**
   * @brief A function to allocate GPU memory
   * @param dev_ptr: the pointer to the start address of the allocated memory
   * @param size: the size of the memory wished to be allocated
   *
   * @return whether the memory is successfully allocated
   */
  cudaError_t malloc(void** dev_ptr, std::size_t size);

  /**
   * @brief A tail-recursive function to spilt a block into two buddies, until
   * the target order is reached
   * @param old_block: the block to be split
   * @param target_order: the order of the block wished to be split into
   *
   * @return the pointer to the start address of the block of the target order
   */
  void* split_buddy(block_info* old_block, std::size_t target_order);

  /**
   * @brief A function to free GPU memory
   * @param dev_ptr: the pointer to the start address of the memory wished to be
   * freed
   *
   * @return whether the memory is successfully freed
   */
  cudaError_t free(void* dev_ptr);

  /**
   * @brief A tail-recursive function to merge a block with its buddy
   * @param block: the block to be merged
   */
  void merge_buddy(block_info* block);

  /**
   * @brief A function to print the free lists
   */
  void print_free_lists();
};

C10_CUDA_API extern std::atomic<GPUMemAllocator*> gpu_mem_allocator;
C10_CUDA_API extern std::atomic<GPUMemAllocator*> gpu_mem_robust_allocator;

inline GPUMemAllocator* get_gpu_mem_allocator() {
  return gpu_mem_allocator.load();
}

inline GPUMemAllocator* get_gpu_robust_mem_allocator() {
  return gpu_mem_robust_allocator.load();
}

inline cudaError_t cudaMalloc_custom(void** dev_ptr, std::size_t size) {
  return get_gpu_mem_allocator()->malloc(dev_ptr, size);
}

inline cudaError_t cudaFree_custom(void* dev_ptr) {
  return get_gpu_mem_allocator()->free(dev_ptr);
}

inline cudaError_t cudaMalloc_robust(void** dev_ptr, std::size_t size) {
  return get_gpu_robust_mem_allocator()->malloc(dev_ptr, size);
}

inline cudaError_t cudaFree_robust(void* dev_ptr) {
  return get_gpu_robust_mem_allocator()->free(dev_ptr);
}

} // namespace cuda::GPUMemAllocator

} // namespace c10