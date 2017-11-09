/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

//#ifdef TENSORFLOW_USE_VERBS

#include "tensorflow/contrib/verbs/rdma_rendezvous_mgr.h"
#include "tensorflow/contrib/verbs/rdma_memory_mgr.h"
#include <unordered_set>
#include "tensorflow/contrib/verbs/verbs_util.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/gpu/gpu_util.h"
#include "tensorflow/core/common_runtime/gpu/process_state.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {

class RdmaRemoteRendezvous : public BaseRemoteRendezvous {
 public:
  RdmaRemoteRendezvous(const WorkerEnv* env, int64 step_id, RdmaMgr* rdma_mgr)
      : BaseRemoteRendezvous(env, step_id), rdma_mgr_(rdma_mgr) {}

 protected:
  void RecvFromRemoteAsync(const Rendezvous::ParsedKey& parsed,
                           const Rendezvous::Args& args,
                           DoneCallback done) override;

 private:
  ~RdmaRemoteRendezvous() override {}
  RdmaMgr* rdma_mgr_;

  TF_DISALLOW_COPY_AND_ASSIGN(RdmaRemoteRendezvous);
};

class DelegatedAllocator: public Allocator {
 public:
//  DelegatedAllocator(Allocator* sub_allocator, void* addr)
//      : sub_allocator_(sub_allocator),
//        delegate_value_(addr) {
//  }

  DelegatedAllocator(Allocator* sub_allocator)
      : sub_allocator_(sub_allocator),
        value_(nullptr) {
  }

  string Name() override {
    std::stringstream s;
    s << "DelegatedAllocator (" << sub_allocator_->Name() << " ==> 0x" << std::hex << value_ << ")";
    return s.str();
  }
  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    if (value_ == nullptr) {
      value_ = sub_allocator_->AllocateRaw(alignment, num_bytes);
    }
    return value_;
  }
  void DeallocateRaw(void* ptr) override {
//    LOG(INFO) << "ELAD: DEALLOCATING " << ptr << ". BASE: " << base_;
    sub_allocator_->DeallocateRaw(value_);
    //delete this;
  }

  void* Value() {
    return value_;
  }
 private:
  Allocator* sub_allocator_;
  void* value_;
};


#define MAX_TENSOR_SIZE         ((10 * 1024 * 1024))

const TensorShape& InitDummyShape() {
  static TensorShape instance;
  instance.AddDim(10);
  instance.AddDim(1024);
  instance.AddDim(1024);
  return instance;
}

const TensorShape& kDummyShape = InitDummyShape();

void RdmaRemoteRendezvous::RecvFromRemoteAsync(
    const Rendezvous::ParsedKey& parsed, const Rendezvous::Args& recv_args,
    DoneCallback done) {
  Status s;
  // parse src_name and dst_name
  string src_name, dst_name, unused;
  if (!DeviceNameUtils::SplitDeviceName(parsed.src_device, &src_name,
                                        &unused)) {
    s = errors::Internal("Could not parse src name.");
  }
  CHECK(s.ok()) << "s is not ok, error code " << s.error_message();
  if (!s.ok()) {
    done(s, Args(), recv_args, Tensor{}, false);
    return;
  }
  if (!DeviceNameUtils::SplitDeviceName(parsed.dst_device, &dst_name,
                                        &unused)) {
    s = errors::Internal("Could not parse dst name.");
  }
  CHECK(s.ok()) << "s is not ok, error code " << s.error_message();
  if (!s.ok()) {
    done(s, Args(), recv_args, Tensor{}, false);
    return;
  }
  CHECK(dst_name.compare(rdma_mgr_->local_worker()) == 0);
  RdmaChannel* rc = rdma_mgr_->FindChannel(src_name);
  string key(std::move(parsed.FullKey().ToString()));
  string key_with_step_id = VerbsUtil::AppendStepidToKey(key, step_id_);
  uint64_t step_id = step_id_;

  Device* src_dev;
  s = env_->device_mgr->LookupDevice("CPU:0", &src_dev);
  CHECK(s.ok()) << "s is not ok, error code " << s.error_message();
  if (!s.ok()) {
    done(s, Args(), recv_args, Tensor(), true);
    return;
  }

  Device* dst_dev;
  s = env_->device_mgr->LookupDevice(parsed.dst_device, &dst_dev);
  CHECK(s.ok()) << "s is not ok, error code " << s.error_message();
  if (!s.ok()) {
    done(s, Args(), recv_args, Tensor(), true);
    return;
  }

  //*************************************************************************************
  // Preallocate the result Tensor.
  // If we can't RDMA directly into the result Tensor (no GDR), allocate an RDMA tensor to
  // do rdma_write into, and afterwards do device copy from it to the result Tensor.
  // Either way, we need to know the Tensor size at this stage.
  //*************************************************************************************

  DelegatedAllocator* result_tensor_allocator =
      new DelegatedAllocator(dst_dev->GetAllocator(recv_args.alloc_attrs));
  DelegatedAllocator* rdma_proxy_allocator = nullptr;

  Tensor* result_tensor = nullptr;
  Tensor *proxy_tensor = nullptr;
  size_t tensor_size = 0;
  void* rdma_addr = nullptr;
  ibv_mr* mr = nullptr;

  const TensorMetaData* meta_data = RdmaMemoryMgr::Singleton().GetTensorMetaData(key);
  if (meta_data == nullptr)
  {
    tensor_size = 0xa0000;
    rdma_addr = result_tensor_allocator->AllocateRaw(32, tensor_size);
  }
  else
  {
    result_tensor = new Tensor(result_tensor_allocator,
                               meta_data->data_type_,
                               meta_data->tensor_shape_);
    tensor_size = result_tensor->TotalBytes();
    rdma_addr = result_tensor_allocator->Value();
  }

  if (tensor_size > 0 )
  {
    mr = RdmaMemoryMgr::Singleton().FindMemoryRegion(rdma_addr, tensor_size);
    if (mr == nullptr)
    {
      // Can't RDMA directly to result. Use a proxy.
      rdma_proxy_allocator = new DelegatedAllocator(ProcessState::singleton()->GetCUDAHostAllocator(0));
      if (meta_data == nullptr)
      {
        rdma_addr = rdma_proxy_allocator->AllocateRaw(32, tensor_size);
      }
      else
      {
        proxy_tensor = new Tensor(rdma_proxy_allocator,
                                  meta_data->data_type_,
                                  meta_data->tensor_shape_);
        rdma_addr = rdma_proxy_allocator->Value();
      }
      mr = RdmaMemoryMgr::Singleton().FindMemoryRegion(rdma_addr, tensor_size);
    }

    if (mr == nullptr) {
      LOG(FATAL) << "ELAD COULD NOT GET AN RDMABLE DESTINATION ADDRESS"
                 << ".\n RESULT: " << result_tensor_allocator->Name()
                 << ".\n PROXY: "  << rdma_proxy_allocator->Name()
                 << ".\n SIZE: "   << tensor_size;
    }
  }

  int pending_request_index = rc->PutPendingRequest(rdma_addr);

  // insert callback
  // ELAD: CALLBACK TO BE DONE WHEN RECEIVING RDMA_MESSAGE_TENSOR_WRITE.
  //        Create the result tensor from the written data, and invoke the DoneCallback.

  rc->InsertRecvCallback(pending_request_index, [this, key, step_id, rc, pending_request_index,
                                                 rdma_addr, result_tensor, proxy_tensor,
                                                 result_tensor_allocator, rdma_proxy_allocator,
                                                 src_dev, dst_dev, recv_args, parsed, done]() {

    Status s;
    Tensor* rt = result_tensor;
    Tensor* pt = proxy_tensor;
    if (rt == nullptr) {

      const TensorMetaData* mt = RdmaMemoryMgr::Singleton().GetTensorMetaData(key);
      while (mt == nullptr) {
//        LOG(INFO) << "STEP 0x" << std::hex << step_id << ": WAITING FOR TENSOR " << key << " META-DATA";
        mt = RdmaMemoryMgr::Singleton().GetTensorMetaData(key);
      }

       rt = new Tensor(result_tensor_allocator,
                       mt->data_type_,
                       mt->tensor_shape_);

       if (rdma_proxy_allocator != nullptr)
       {
         pt = new Tensor(rdma_proxy_allocator,
                         mt->data_type_,
                         mt->tensor_shape_);
       }
    }


//    LOG(INFO) << "STEP 0x" << std::hex << step_id << std::dec
//              << ": RECEIVED RESPONSE #" << pending_request_index << ": " << key << ": " //<< result_tensor->DebugString()
//              << "(SIZE: 0x" << std::hex << rt->TotalBytes() << ")";

    Tensor val;
    if (pt != nullptr)
      /*dst_dev->tensorflow_gpu_device_info() &&
        (!recv_args.alloc_attrs.on_host()))*/ {

//      LOG(INFO) << "ELAD PROXYING " << std::hex << rdma_proxy_allocator->Name() << " ==> " << result_tensor_allocator->Name();

      GPUUtil::CopyCPUTensorToGPU(
          pt, recv_args.device_context, dst_dev, rt,
          [this, pt, rt, pending_request_index, recv_args, done, rc](const Status& s) {
            CHECK(s.ok()) << "copy tensor to gpu sync";
            Tensor val;
            val = std::move(*rt);
            delete rt;
            delete pt;
            rc->RemoveRecvCallback(pending_request_index);
            done(s, Args(), recv_args, val, false);
          });
      return;
    }
    val = std::move(*rt);
    delete rt;
    rc->RemoveRecvCallback(pending_request_index);
    done(s, Args(), recv_args, val, false);
  });

  // append key to message queue
  RdmaMessage rm;
  rm.type_ = RDMA_MESSAGE_TENSOR_REQUEST;
  rm.tensor_bytes_ = (RdmaMessageType)pending_request_index;
  rm.name_size_ = key.size();
  rm.name_ = key;
  rm.step_id_ = step_id_;
  rm.remote_addr_ = (uint64_t)rdma_addr;
  rm.rkey_ = (mr == nullptr) ? 0 : mr->rkey;

//  LOG(INFO) << "STEP 0x" << std::hex << rm.step_id_ << std::dec
//            << ": SENDING REQUEST #" << pending_request_index << ": " << rm.name_
//            << " ON " << result_tensor_allocator->Name() << " (RKEY: 0x" << std::hex << rm.rkey_ << ")";

  string message = RdmaMessage::CreateMessage(rm);
  rc->tx_message_buffer_->EnqueueItem(message);
  rc->tx_message_buffer_->SendNextItem();
}

RdmaRendezvousMgr::RdmaRendezvousMgr(const WorkerEnv* env)
    : BaseRendezvousMgr(env) {}

BaseRemoteRendezvous* RdmaRendezvousMgr::Create(int64 step_id,
                                                const WorkerEnv* worker_env) {
  return new RdmaRemoteRendezvous(worker_env, step_id, rdma_mgr_);
}

}  // end namespace tensorflow

//#endif
