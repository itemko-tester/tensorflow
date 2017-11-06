/*
 * RdmaMemoryMgr.h
 *
 *  Created on: Nov 5, 2017
 *      Author: eladw
 */

#ifndef TENSORFLOW_CONTRIB_VERBS_RDMA_MEMORY_MGR_H_
#define TENSORFLOW_CONTRIB_VERBS_RDMA_MEMORY_MGR_H_

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <infiniband/verbs.h>

#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {


class Device;
class DeviceContext;
class Tensor;

void MRDeleter(ibv_mr* mr);

using MemoryRegionPtr = std::unique_ptr<ibv_mr, decltype(&MRDeleter)>;

using TensorAllocationMetaData = std::pair<DataType, TensorShape>;

class RdmaMemoryMgr {

  public:
    static RdmaMemoryMgr& Singleton() { static RdmaMemoryMgr instance; return instance; }

    // Memory regions
    ibv_mr* FindMemoryRegion(void* addr, size_t length);
    void InsertMemoryRegion(void* addr, size_t length);
    void EvictMemoryRegion(void* addr, size_t length);

    const int GetTensorSize(std::string tensor_name) {
      mutex_lock l(tensor_sizes_mu_);
      auto it = tensor_sizes_.find(tensor_name);
      if (it == tensor_sizes_.end()) {
        return -1;
      }
      return it->second;
    }

    void SetTensorSize(std::string tensor_name, int size) {
      mutex_lock l(tensor_sizes_mu_);
      tensor_sizes_[tensor_name] = size;
    }

    struct ibv_pd *pd_;

 protected:
    RdmaMemoryMgr() { }

  static bool Comparator(const void* ptr, const MemoryRegionPtr& other) {
    return ptr < reinterpret_cast<char*>(other->addr) + other->length;
  }

 private:
  mutex tensor_sizes_mu_;
  std::unordered_map<std::string, int> tensor_sizes_;

  // Managed memory regions
  mutex alloc_mu_;
  std::vector<MemoryRegionPtr> mrs_ GUARDED_BY(alloc_mu_);

};

}

#endif /* TENSORFLOW_CONTRIB_VERBS_RDMA_MEMORY_MGR_H_ */
