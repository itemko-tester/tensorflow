/*
 * RdmaMemoryMgr.cpp
 *
 *  Created on: Nov 5, 2017
 *      Author: eladw
 */

#include <fcntl.h>
#include <fstream>

#include "rdma_memory_mgr.h"
#include "tensorflow/core/framework/allocator_registry.h"

#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"

#include "tensorflow/core/common_runtime/bfc_allocator.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/gpu/gpu_util.h"
#include "tensorflow/core/common_runtime/gpu/process_state.h"

namespace tensorflow {


ibv_mr* RdmaMemoryMgr::FindMemoryRegion(void* addr, size_t length) {
  mutex_lock l(alloc_mu_);
  auto iter = std::upper_bound(mrs_.begin(), mrs_.end(), addr, &Comparator);
  if (iter == std::end(mrs_) || iter->get()->addr > addr) {
    return nullptr;
  } else {
    return iter->get();
  }
}

void RdmaMemoryMgr::InsertMemoryRegion(void* addr, size_t length) {
  if (length == 0) return;
  ibv_mr* mr = ibv_reg_mr(pd_, addr, length,
                     IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
//  LOG(INFO) << "INSERT MEMORY REGION 0x" << std::hex << mr->rkey << ". ADDRESS: 0x" << addr << ". SIZE: 0x" << length;
  if (mr != nullptr) {
    mutex_lock l(alloc_mu_);
    auto iter = std::upper_bound(mrs_.begin(), mrs_.end(), addr, &Comparator);
    mrs_.insert(iter, {mr, &MRDeleter});
  } else {
    LOG(WARNING) << "Cannot register memory region";
  }
}

void RdmaMemoryMgr::EvictMemoryRegion(void* addr, size_t length) {
  if (length == 0) return;
  mutex_lock l(alloc_mu_);
  auto iter = std::upper_bound(mrs_.begin(), mrs_.end(), addr, &Comparator);
  if (iter != std::end(mrs_) && iter->get()->addr == addr) {
    mrs_.erase(iter);
    LOG(INFO) << "EVICT MEMORY REGION 0x" << std::hex << iter->get()->rkey;

  } else {
    LOG(WARNING) << "Failed to de-register memory region";
  }
}

}
