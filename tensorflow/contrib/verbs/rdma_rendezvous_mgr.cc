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

#ifdef TENSORFLOW_USE_VERBS

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

  void RecvPostCopyOps(const string& key, const string& key_with_step_id,
                       const Rendezvous::Args& recv_args,
                       const DoneCallback& done, const RdmaMessage& rm,
                       RdmaChannel* rc, Tensor& val, const Status& s);

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

  DelegatedAllocator(Allocator* sub_allocator,
                     size_t alignment, size_t num_bytes,
                     size_t offset)
      : sub_allocator_(sub_allocator),
        base_(sub_allocator->AllocateRaw(alignment, num_bytes)),
        offset_(offset) {
  }

  string Name() override {
    std::stringstream s;
    s << "DelegatedAllocator (" << sub_allocator_->Name() << " ==> 0x" << std::hex << base_ << " + 0x" << offset_ << ")";
    return s.str();
  }
  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    return Value();
  }
  void DeallocateRaw(void* ptr) override {
//    LOG(INFO) << "ELAD: DEALLOCATING " << ptr << ". BASE: " << base_;
    sub_allocator_->DeallocateRaw(base_);
    //delete this;
  }
  void* Base() {
    return base_;
  }
  size_t Offset() {
    return offset_;
  }
  void* Value() {
    return base_ + offset_;
  }
 private:
  Allocator* sub_allocator_;
  void* base_;
  size_t offset_;
};


#define MAX_TENSOR_SIZE         ((10 * 1024 * 1024))

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

  int tensor_size = RdmaMemoryMgr::Singleton().GetTensorSize(key);
  if (tensor_size == -1) {
    tensor_size = MAX_TENSOR_SIZE;
  }

  DelegatedAllocator* result_tensor_allocator =
      new DelegatedAllocator(dst_dev->GetAllocator(recv_args.alloc_attrs),
                             EIGEN_MAX_ALIGN_BYTES,
                             RdmaMessage::kTensorBufferStartIndex + tensor_size,
                             RdmaMessage::kTensorBufferStartIndex);

  void* rdma_addr = result_tensor_allocator->Base();
  ibv_mr* mr = RdmaMemoryMgr::Singleton().FindMemoryRegion(rdma_addr, tensor_size);
  DelegatedAllocator* rdma_proxy_allocator = nullptr;

  if (mr == nullptr) {
    // Can't RDMA directly to result. Use a proxy.
    rdma_proxy_allocator =
        new DelegatedAllocator(ProcessState::singleton()->GetCUDAHostAllocator(0),
                               EIGEN_MAX_ALIGN_BYTES,
                               RdmaMessage::kTensorBufferStartIndex + tensor_size,
                               RdmaMessage::kTensorBufferStartIndex);
    rdma_addr = rdma_proxy_allocator->Base();
    mr = RdmaMemoryMgr::Singleton().FindMemoryRegion(rdma_addr, tensor_size);
  }

  if (mr == nullptr) {
    LOG(FATAL) << "ELAD COULD NOT GET AN RDMABLE DESTINATION ADDRESS"
               << ".\n RESULT: " << result_tensor_allocator->Name()
               << ".\n PROXY: "  << rdma_proxy_allocator->Name()
               << ".\n SIZE: "   << tensor_size + RdmaMessage::kTensorBufferStartIndex;
  }

  // insert callback
  // ELAD: CALLBACK TO BE DONE WHEN RECEIVING RDMA_MESSAGE_TENSOR_WRITE.
  //        Create the result tensor from the written data, and invoke the DoneCallback.

  rc->InsertRecvCallback(key_with_step_id, [this, key, key_with_step_id, rc,
                                            rdma_addr, result_tensor_allocator, rdma_proxy_allocator,
                                            src_dev, dst_dev, recv_args, parsed, done]() {
    Status s;
    RdmaMessage rm;
    RdmaMessage::ParseMessage(rm, rdma_addr);

    RdmaMemoryMgr::Singleton().SetTensorSize(key, rm.buffer_size_);

    Tensor* result_tensor = new Tensor(
        result_tensor_allocator,
        rm.data_type_,
        rm.tensor_shape_);

    LOG(INFO) << "ELAD: RECEIVED TENSOR RESPONSE " << result_tensor_allocator->Name();
    return;

    Tensor val;
    if (!rm.is_dead_) {

      bool can_memcpy = DataTypeCanUseMemcpy(rm.data_type_);
      if (can_memcpy) {
        if (rdma_proxy_allocator != nullptr)
          /*dst_dev->tensorflow_gpu_device_info() &&
            (!recv_args.alloc_attrs.on_host()))*/ {

          LOG(INFO) << "ELAD PROXYING " << std::hex << rdma_proxy_allocator->Name() << " ==> " << result_tensor_allocator->Name();
          CHECK(recv_args.device_context)
            << "send dev name: " << src_dev->name()
            << " gpu_info: " << src_dev->tensorflow_gpu_device_info();

          Tensor proxy(rdma_proxy_allocator, rm.data_type_, rm.tensor_shape_);
          GPUUtil::CopyCPUTensorToGPU(
              &proxy, recv_args.device_context, dst_dev, result_tensor,
              [this, result_tensor, key, key_with_step_id, recv_args, done, rm,
               rc](const Status& s) {
                CHECK(s.ok()) << "copy tensor to gpu sync";
                Tensor val;
                val = std::move(*result_tensor);
                delete result_tensor;
                RecvPostCopyOps(key, key_with_step_id, recv_args, done, rm, rc,
                                val, s);
              });
          return;
        }
      } else {
        TensorProto proto;
        CHECK(ParseProtoUnlimited(&proto, result_tensor_allocator->Value(), rm.tensor_bytes_))
            << "fail to parse proto from array";
        s = dst_dev->MakeTensorFromProto(proto, recv_args.alloc_attrs, &val);
      }
    }
    val = std::move(*result_tensor);
    delete result_tensor;
    RecvPostCopyOps(key, key_with_step_id, recv_args, done, rm, rc, val, s);
  });
  // append key to message queue
  int pending_request_index = rc->PutPendingRequest(rdma_addr);

  RdmaMessage rm;
  rm.type_ = (RdmaMessageType)pending_request_index;
  rm.name_size_ = key.size();
  rm.name_ = key;
  rm.step_id_ = step_id_;
  rm.remote_addr_ = (uint64_t)rdma_addr;
  rm.rkey_ = mr->rkey;

  LOG(INFO) << "ELAD: SENDING REQUEST #" << pending_request_index << ". DST_ADDR: " << rdma_addr << " RKEY: 0x" << std::hex << rm.rkey_ << ". TENSOR-SIZE: 0x" << tensor_size << ". TENSOR: " << rm.name_;

  string message = RdmaMessage::CreateMessage(rm);
  rc->tx_message_buffer_->EnqueueItem(message);
  rc->tx_message_buffer_->SendNextItem();
}

void RdmaRemoteRendezvous::RecvPostCopyOps(
    const string& key, const string& key_with_step_id,
    const Rendezvous::Args& recv_args, const DoneCallback& done,
    const RdmaMessage& rm, RdmaChannel* rc, Tensor& val, const Status& s) {

  rc->RemoveRecvCallback(key_with_step_id);
//  RdmaMessage br;
//  br.type_ = RDMA_MESSAGE_BUFFER_IDLE;
//  br.name_size_ = key.size();
//  br.name_ = key;
//  string message = RdmaMessage::CreateMessage(br);
//  RdmaBuffer* tb = rc->tx_message_buffer_;
//  tb->EnqueueItem(message);
//  tb->SendNextItem();
  done(s, Args(), recv_args, val, rm.is_dead_);
}

RdmaRendezvousMgr::RdmaRendezvousMgr(const WorkerEnv* env)
    : BaseRendezvousMgr(env) {}

BaseRemoteRendezvous* RdmaRendezvousMgr::Create(int64 step_id,
                                                const WorkerEnv* worker_env) {
  return new RdmaRemoteRendezvous(worker_env, step_id, rdma_mgr_);
}

}  // end namespace tensorflow

#endif
