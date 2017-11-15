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


  void SendTensorMetaDataRequest(RdmaChannel* rc,
                                 RdmaTensorRequest* request,
                                 uint64_t step_id,
                                 const Rendezvous::ParsedKey& parsed,
                                 const Rendezvous::Args& recv_args,
                                 DoneCallback done);

  void SendTensorContentRequest(RdmaChannel* rc,
                                RdmaTensorRequest* request,
                                uint64_t step_id,
                                const Rendezvous::ParsedKey& parsed,
                                const Rendezvous::Args& recv_args,
                                const TensorMetaData* meta_data,
                                DoneCallback done);

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
//      debug_mu_.lock();
//      allocated++;
//      debug_mu_.unlock();
    }
    return value_;
  }
  void DeallocateRaw(void* ptr) override {
//    debug_mu_.lock();
//    deallocated++;
//    debug_mu_.unlock();
//    if (deallocated % 1000 == 0) {
//      LOG(INFO) << "ALLOCATED: " << allocated << ". DEALLOCATED: " << deallocated;
//    }

    CHECK(ptr == value_);
    sub_allocator_->DeallocateRaw(ptr);
    //delete this;
  }

  void* Value() {
    return value_;
  }
 private:
//  static mutex debug_mu_;
//  static int allocated;
//  static int deallocated;
  Allocator* sub_allocator_;
  void* value_;
};

//mutex DelegatedAllocator::debug_mu_;
//int DelegatedAllocator::allocated = 0;
//int DelegatedAllocator::deallocated = 0;

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
  RdmaTensorRequest* request = rc->CreatePendingRequest();


  //*************************************************************************************
  // Preallocate the result Tensor.
  // If we can't RDMA directly into the result Tensor (no GDR), allocate an RDMA tensor to
  // do rdma_write into, and afterwards do device copy from it to the result Tensor.
  // Either way, we need to know the Tensor size at this stage.
  //*************************************************************************************

  string key(std::move(parsed.FullKey().ToString()));
  const TensorMetaData* meta_data = RdmaMemoryMgr::Singleton().GetTensorMetaData(key);
  if (meta_data == nullptr)
  {
    SendTensorMetaDataRequest(rc, request, step_id_, parsed, recv_args, done);
  }
  else
  {
    SendTensorContentRequest(rc, request, step_id_, parsed, recv_args, meta_data, done);
  }
}

void RdmaRemoteRendezvous::SendTensorMetaDataRequest(RdmaChannel* rc,
                                                     RdmaTensorRequest* request,
                                                     uint64_t step_id,
                                                     const Rendezvous::ParsedKey& parsed,
                                                     const Rendezvous::Args& recv_args,
                                                     DoneCallback done)
{
  string key(std::move(parsed.FullKey().ToString()));

  RdmaMessage rm;
  rm.type_ = RDMA_MESSAGE_TENSOR_META_DATA_REQUEST;
  rm.name_size_ = key.size();
  rm.name_ = key;
  rm.step_id_ = step_id;
  rm.request_index_ = request->index_;

  RdmaTensorRequest::RecvMetaDataCallback* cb = new RdmaTensorRequest::RecvMetaDataCallback(
      [this, rc, parsed, recv_args, request, step_id, key, done](const DataType& dtype, const TensorShape& shape)
      {
          TensorMetaData* meta_data = RdmaMemoryMgr::Singleton().SetTensorMetaData(key,
                                                                                   dtype,
                                                                                   shape);
          SendTensorContentRequest(rc, request, step_id, parsed, recv_args, meta_data, done);
      }
  );

  request->recv_meta_data_callback_ = cb;

//  LOG(INFO) << "STEP 0x" << std::hex << rm.step_id_ << std::dec
//            << ": SENDING META-DATA REQUEST #" << request->index_ << ": " << rm.name_;

  string message = RdmaMessage::CreateMessage(rm);
  rc->tx_message_buffer_->EnqueueItem(message);
  rc->tx_message_buffer_->SendNextItem();
}

void RdmaRemoteRendezvous::SendTensorContentRequest(RdmaChannel* rc,
                                                    RdmaTensorRequest* request,
                                                    uint64_t step_id,
                                                    const Rendezvous::ParsedKey& parsed,
                                                    const Rendezvous::Args& recv_args,
                                                    const TensorMetaData* meta_data,
                                                    DoneCallback done)
{
  Status s;
  static uint64_t num_proxies = 0;
  static uint64_t num_proxy_bytes = 0;
  // parse src_name and dst_name

  string key(std::move(parsed.FullKey().ToString()));
  Device* dst_dev;
  s = env_->device_mgr->LookupDevice(parsed.dst_device, &dst_dev);
  CHECK(s.ok()) << "s is not ok, error code " << s.error_message();
  if (!s.ok()) {
    done(s, Args(), recv_args, Tensor(), true);
    return;
  }

  DelegatedAllocator* result_tensor_allocator =
      new DelegatedAllocator(dst_dev->GetAllocator(recv_args.alloc_attrs));
  DelegatedAllocator* rdma_proxy_allocator = nullptr;

  Tensor* result_tensor = nullptr;
  Tensor *proxy_tensor = nullptr;
  size_t tensor_size = 0;
  void* rdma_addr = nullptr;
  ibv_mr* mr = nullptr;


  result_tensor = new Tensor(result_tensor_allocator,
                             meta_data->data_type_,
                             meta_data->tensor_shape_);
  tensor_size = result_tensor->TotalBytes();
  rdma_addr = result_tensor_allocator->Value();

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

  // insert callback
  // ELAD: CALLBACK TO BE DONE WHEN RECEIVING RDMA_MESSAGE_TENSOR_WRITE.
  //        Create the result tensor from the written data, and invoke the DoneCallback.

  RdmaTensorRequest::RecvContentCallback* cb = new RdmaTensorRequest::RecvContentCallback(
      [this, key, step_id, rc, request,
       rdma_addr, result_tensor, proxy_tensor,
       result_tensor_allocator, rdma_proxy_allocator,
       dst_dev, recv_args, parsed, done]()
  {
    Status s;
//    LOG(INFO) << "STEP 0x" << std::hex << step_id << std::dec
//              << ": RECEIVED CONTENT RESPONSE #" << request->index_ << ": " << key << ": " //<< result_tensor->DebugString()
//              << "(SIZE: 0x" << std::hex << result_tensor->TotalBytes() << ")";

    Tensor val;
    if (proxy_tensor != nullptr)
      /*dst_dev->tensorflow_gpu_device_info() &&
        (!recv_args.alloc_attrs.on_host()))*/ {

//      LOG(INFO) << "ELAD PROXYING " << std::hex << rdma_proxy_allocator->Name() << " ==> " << result_tensor_allocator->Name();
      num_proxy_bytes += result_tensor->TotalBytes();
      if (++num_proxies % 1000 == 0) {
        LOG(INFO) << "Number of proxies: " << num_proxies << " (" << num_proxy_bytes << " Bytes).";
      }

      GPUUtil::CopyCPUTensorToGPU(
          proxy_tensor, recv_args.device_context, dst_dev, result_tensor,
          [this, proxy_tensor, result_tensor, request, recv_args, done, rc](const Status& s) {
            CHECK(s.ok()) << "copy tensor to gpu sync";
            Tensor val;
            val = std::move(*result_tensor);
            delete result_tensor;
            delete proxy_tensor;
            done(s, Args(), recv_args, val, false);
          });
      return;
    }

    val = std::move(*result_tensor);
    delete result_tensor;

//
//    static int count = 0;
//    if (count++ == 2) {
//      return;
//    }

    done(s, Args(), recv_args, val, false);
  });

  request->recv_content_callback_ = cb;

  // append key to message queue
  RdmaMessage rm;
  rm.type_ = RDMA_MESSAGE_TENSOR_CONTENT_REQUEST;
  rm.request_index_ = request->index_;
  rm.name_size_ = key.size();
  rm.name_ = key;
  rm.step_id_ = step_id_;
  rm.remote_addr_ = (uint64_t)rdma_addr;
  rm.rkey_ = (mr == nullptr) ? 0 : mr->rkey;

//  LOG(INFO) << "STEP 0x" << std::hex << rm.step_id_ << std::dec
//            << ": SENDING CONTENT REQUEST #" << request->index_ << ": " << rm.name_
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
