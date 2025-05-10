#include <c10/cuda/GPUMemAllocator.h>

#include <cuda_runtime_api.h>
#include <atomic>
#include <iostream>

namespace c10 {

namespace cuda::GPUMemAllocator {

// initialize a 4GB memory space on GPU
// GPUMemAllocator gpu_mem_allocator_instance(4 * ((std::size_t)1 << 30));
GPUMemAllocator gpu_mem_allocator_instance((std::size_t)1 << 32, false);
GPUMemAllocator gpu_mem_robust_allocator_instance((std::size_t)1 << 32, true);

struct GPUMemAllocatorInitializer {
  GPUMemAllocator* parseEnv() {
    // TODO:

    return &gpu_mem_allocator_instance;
  }

  GPUMemAllocatorInitializer() {
    // auto g = parseEnv();
    gpu_mem_allocator.store(&gpu_mem_allocator_instance);
    gpu_mem_robust_allocator.store(&gpu_mem_robust_allocator_instance);
  }
};

GPUMemAllocatorInitializer gpu_mem_allocator_initializer;

GPUMemAllocator::GPUMemAllocator(std::size_t total_mem, bool is_robust_) : device_ptr(nullptr), is_robust(is_robust_) {
  // Align total_mem to nearest the power of 2, get the order of total_mem
  std::size_t power = 1;
  std::size_t order = 0;
  while (power < total_mem) {
    power <<= 1;
    order++;
  }
  TOTAL_MEM = power;
  MAX_ORDER = order;

  // Alloc a huge chunk of memory on GPU amd initialize free_list
  cudaError_t err = cudaMalloc(&device_ptr, TOTAL_MEM);
  if (err != cudaSuccess) {
    std::cerr << "GPUMemAllocator initialization failed" << std::endl;
    std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
    exit(1);
  }
  free_lists.resize(MAX_ORDER + 1);

  // Initialize the first biggest block
  block_info* init_block = new block_info();
  init_block->dev_ptr = device_ptr;
  init_block->is_free = true;
  init_block->order = MAX_ORDER;
  init_block->pred = nullptr;
  init_block->succ = nullptr;
  free_lists[MAX_ORDER].push_back(init_block);

  block_map[device_ptr] = init_block;

  //...DEBUG...//
  if(is_robust)
    std::cout << "[GPU MEMORY ALLOCATOR INFO]" << "Robust GPUMemAllocator initialized, start address: " << device_ptr
              << ",end address: " << (void*)((char*)device_ptr + TOTAL_MEM) << std::endl;
  else
    std::cout << "[GPU MEMORY ALLOCATOR INFO]" << "Vulnerable GPUMemAllocator initialized, start address: " << device_ptr
              << ",end address: " << (void*)((char*)device_ptr + TOTAL_MEM) << std::endl;
  //...DEBUG...//
}

GPUMemAllocator::~GPUMemAllocator() {
  cudaFree(device_ptr);
  // free every block_info in block_map
  for (auto it = block_map.begin(); it != block_map.end(); it++) {
    delete it->second;
  }
}

cudaError_t GPUMemAllocator::malloc(void** dev_ptr, std::size_t size) {
  // Align size to nearest the power of 2, get the order of size
  std::size_t power = 1;
  std::size_t order = 0;
  std::size_t target_order = 0;
  while (power < size) {
    power <<= 1;
    order++;
    target_order++;
  }

  // Find the first block that can be split into blocks of order 'order'
  while (order <= MAX_ORDER && free_lists[order].empty()) {
    order++;
  }

  // If there is no block that can be split, return nullptr
  if (order > MAX_ORDER) {
    std::cerr << "malloc failed 1" << std::endl;
    std::cerr << "Error: Out of memory" << std::endl;
    return cudaError::cudaErrorMemoryAllocation;
  }

  // Split the block into blocks of order 'order'
  auto block = free_lists[order].front();
  free_lists[order].pop_front();
  block->is_free = false;

  auto ret = split_buddy(block, target_order);
  if (ret == nullptr) {
    // should never reach here, otherwise there is a bug in the code
    std::cerr << "malloc failed 2" << std::endl;
    std::cerr << "Error: Out of memory" << std::endl;
    return cudaError::cudaErrorMemoryAllocation;
  }

  *dev_ptr = ret;

  return cudaError::cudaSuccess;
}

void* GPUMemAllocator::split_buddy(
    block_info* old_block,
    std::size_t target_order) {
  auto current_order = old_block->order;
  if (current_order == target_order) {
    return old_block->dev_ptr;
  }

  auto new_block = new block_info();
  new_block->dev_ptr =
      (void*)(((char*)old_block->dev_ptr) + (1 << (current_order - 1)));
  new_block->is_free = true;
  new_block->order = current_order - 1;
  new_block->pred = old_block;
  new_block->succ = old_block->succ;
  free_lists[current_order - 1].push_front(new_block);
  block_map[new_block->dev_ptr] = new_block;

  if (old_block->succ != nullptr) {
    old_block->succ->pred = new_block;
  }

  old_block->order = current_order - 1;
  old_block->succ = new_block;

  return split_buddy(old_block, target_order);
}

cudaError_t GPUMemAllocator::free(void* ptr) {
  // test if ptr is a valid pointer
  if (block_map.find(ptr) == block_map.end()) {
    std::cerr
        << "Error: try to free a pointer which is not the return value of malloc"
        << std::endl;
    return cudaError::cudaErrorInvalidValue;
  }

  // get the block_info of ptr
  auto block = block_map[ptr];
  if (block->is_free) {
    std::cerr << "Error: try to free a pointer which has been freed"
              << std::endl;
    return cudaError::cudaErrorInvalidValue;
  }

  // free the block
  block->is_free = true;
  free_lists[block->order].push_front(block);

  // merge buddy
  merge_buddy(block);

  return cudaError::cudaSuccess;
}

void GPUMemAllocator::merge_buddy(block_info* block) {
  // We promise that block is in free_lists and block_map
  auto current_order = block->order;
  block_info* buddy = nullptr;
  block_info* pred = nullptr;
  block_info* succ = nullptr;

  if (block->pred != nullptr && block->pred->is_free &&
      block->pred->order == current_order &&
      (void*)((char*)block->pred->dev_ptr + (1 << current_order)) ==
          block->dev_ptr) {
    buddy = block->pred;
    pred = buddy;
    succ = block;
  } else if (
      block->succ != nullptr && block->succ->is_free &&
      block->succ->order == current_order &&
      (void*)((char*)block->dev_ptr + (1 << current_order)) ==
          block->succ->dev_ptr) {
    buddy = block->succ;
    pred = block;
    succ = buddy;
  }

  if (buddy == nullptr) {
    return;
  }

  // remove block and buddy from free_lists and block_map
  free_lists[current_order].remove(block);
  free_lists[current_order].remove(buddy);
  block_map.erase(block->dev_ptr);
  block_map.erase(buddy->dev_ptr);

  pred->order = current_order + 1;
  pred->succ = succ->succ;
  if (succ->succ != nullptr) {
    succ->succ->pred = pred;
  }
  free_lists[current_order + 1].push_front(pred);
  block_map[pred->dev_ptr] = pred;

  delete succ;

  merge_buddy(pred);
}

void GPUMemAllocator::print_free_lists() {
  for (std::size_t i = 0; i <= MAX_ORDER; i++) {
    std::cout << "order " << i << ": ";
    for (auto it = free_lists[i].begin(); it != free_lists[i].end(); it++) {
      std::cout << (*it)->dev_ptr << " ";
    }
    std::cout << std::endl;
  }
}

std::atomic<GPUMemAllocator*> gpu_mem_allocator;
std::atomic<GPUMemAllocator*> gpu_mem_robust_allocator;
} // namespace cuda::GPUMemAllocator

} // namespace c10