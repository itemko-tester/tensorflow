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

class TensorMetaData {
  public:
    TensorMetaData(DataType data_type, TensorShape tensor_shape):
      data_type_(data_type), tensor_shape_(tensor_shape) { }

    DataType data_type_;
    TensorShape tensor_shape_;
};

class RdmaMemoryMgr {

  public:
    static RdmaMemoryMgr& Singleton() { static RdmaMemoryMgr instance; return instance; }

    // Memory regions
    ibv_mr* FindMemoryRegion(void* addr, size_t length);
    void InsertMemoryRegion(void* addr, size_t length, const std::string& allocator_name);
    void EvictMemoryRegion(void* addr, size_t length);

    const TensorMetaData* GetTensorMetaData(const std::string& tensor_name) {
      mutex_lock l(tensor_sizes_mu_);
      auto it = tensors_meta_data_.find(tensor_name);
      if (it == tensors_meta_data_.end()) {
        return nullptr;
      }
      return &it->second;
    }

    // Return true if inserted new
    bool SetTensorMetaData(const std::string& tensor_name, DataType dtype, const TensorShape& shape) {
      mutex_lock l(tensor_sizes_mu_);
      TensorMetaData meta_data(dtype, shape);
      auto res = tensors_meta_data_.insert(std::make_pair(tensor_name, meta_data));
      return res.second;
    }

    struct ibv_pd *pd_;

 protected:
    RdmaMemoryMgr() { }

  static bool Comparator(const void* ptr, const MemoryRegionPtr& other) {
    return ptr < reinterpret_cast<char*>(other->addr) + other->length;
  }

 private:
  mutex tensor_sizes_mu_;
  std::unordered_map<std::string, TensorMetaData> tensors_meta_data_;

  // Managed memory regions
  mutex alloc_mu_;
  std::vector<MemoryRegionPtr> mrs_ GUARDED_BY(alloc_mu_);

};

}

#endif /* TENSORFLOW_CONTRIB_VERBS_RDMA_MEMORY_MGR_H_ */
