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

#include "tensorflow/contrib/ucx/ucx_rendezvous_mgr.h"
#include <unordered_set>
#include "tensorflow/core/framework/tensor.h"
//#include "tensorflow/contrib/verbs/verbs_util.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/gpu/gpu_util.h"
#include "tensorflow/core/common_runtime/gpu/process_state.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {

void UcxRemoteRendezvous::RecvFromRemoteAsync(
    const Rendezvous::ParsedKey& parsed, const Rendezvous::Args& recv_args,
    DoneCallback done) {
  Status s;
  // parse src_name and dst_name
  string src_name, dst_name, unused;

  if (!DeviceNameUtils::SplitDeviceName(parsed.src_device, &src_name,
                                        &unused)) {
    s = errors::Internal("Could not parse src name.");
  }
  if (!s.ok()) {
    done(s, Args(), recv_args, Tensor{}, false);
    return;
  }
  if (!DeviceNameUtils::SplitDeviceName(parsed.dst_device, &dst_name,
                                        &unused)) {
    s = errors::Internal("Could not parse dst name.");
  }

  if (!s.ok()) {
    done(s, Args(), recv_args, Tensor{}, false);
    return;
  }

  //*************************************************************************************
  // Need to understand and implement the functions:
  //*************************************************************************************

  // CHECK(dst_name.compare(ucx_mgr_->local_worker()) == 0);

  // ecp_ep_h ep;
  // ep = get_ep / ep is an input param
  // UcxChannel* rc = rdma_mgr_->FindChannel(src_name);

  //  string key(std::move(parsed.FullKey().ToString()));
  //  const TensorMetaData* meta_data = nullptr;//GetTensorMetaData(key);
  //
  //  if (meta_data == nullptr) {
  //    RecvTensorMetaData();
  //  }
  //  else {
  //    RecvTensorContent();
  //  }
}

#if 0
void UcxRemoteRendezvous::RecvMetaData(const Rendezvous::ParsedKey& parsed,
                                       const Rendezvous::Args& recv_args,
                                       const TensorMetaData* meta_data,
                                       Tensor val,
                                       const std::string& tensor_name_meta_data,
                                       std::function<void()> RecvMetaDataCallback,
                                       DoneCallback done) {
  //TODO: implement UCX receive function
  UcxRecv();
  RecvMetaDataCallback(parsed,
                       recv_args,
                       meta_data,
                       val,
                       done);
}
#endif

#if 0
void RecvTensorContent(const Rendezvous::ParsedKey& parsed,
                       const Rendezvous::Args& recv_args,
                       const TensorMetaData* meta_data,
                       DoneCallback done
                       /*TODO params*/) {
  Status s;
  Tensor *proxy_tensor = nullptr;
  size_t tensor_size = 0;
  void* rdma_addr = nullptr;
  ibv_mr* mr = nullptr;
  Device* dst_dev;
  Allocator* rdma_proxy_allocator = nullptr;

  s = env_->device_mgr->LookupDevice(parsed.dst_device, &dst_dev);
  if (!s.ok()) {
    done(s, Args(), recv_args, Tensor(), true);
    return;
  }

  Tensor* result_tensor = new Tensor(dst_dev->GetAllocator(recv_args.alloc_attrs),
                                    meta_data->data_type_,
                                    meta_data->tensor_shape_);

  bool can_memcpy = DataTypeCanUseMemcpy(meta_data->data_type_);
  if (can_memcpy)
  {
    if (tensor_size > 0){
      dest_addr = DMAHelper::base(result_tensor);
      //TODO: implement FindMemoryRegion for ucx
      mr = UcxMemoryMgr::Singleton().FindMemoryRegion(rdma_addr, tensor_size);
      if (mr == nullptr) {
        // Can't RDMA directly to result. Use a proxy.
        rdma_proxy_allocator = ProcessState::singleton()->GetCUDAHostAllocator(0);
        proxy_tensor = new Tensor(rdma_proxy_allocator,
                                  meta_data->data_type_,
                                  meta_data->tensor_shape_);
        rdma_addr = rdma_proxy_allocator->Value();
      }
    }
    else {
      rdma_addr = 0;
      done(s, Args(), recv_args, Tensor(), true);
      return;
    }
  }
  else {
    rdma_addr = ProcessState::singleton()->GetCPUAllocator(0)->AllocateRaw(32, /*TODO: add:*/meta_data->proto_size);
    mr = RdmaMemoryMgr::Singleton().FindMemoryRegion(rdma_addr, /*TODO: add:*/meta_data->proto_size);
   }

  //TODO: implement UCX receive function
  UcxRecv();

}
#endif

#if 0
void UcxRemoteRendezvous::RecvContentCallback (const Rendezvous::Args& recv_args,
                                               const Rendezvous::ParsedKey& parsed,
                                               bool can_memcpy,
                                               void* rdma_addr,
                                               request,
                                               Tensor* result_tensor,
                                               Tensor* proxy_tensor,
                                               Device* dst_dev,
                                               DoneCallback done) {
  Status s;
  Tensor val;

  if (proxy_tensor != nullptr) {
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

  if (can_memcpy)
  {
    val = std::move(*result_tensor);
    delete result_tensor;
  }
  else
  {
    int proto_size = *(int*)rdma_addr;
    TensorProto proto;
    CHECK(ParseProtoUnlimited(&proto, rdma_addr + 4, proto_size)) << "fail to parse proto from array";
    s = dst_dev->MakeTensorFromProto(proto, recv_args.alloc_attrs, &val);
  }
  done(s, Args(), recv_args, val, false);
}



void UcxRemoteRendezvous::RecvMetaDataCallback(const Rendezvous::ParsedKey& parsed,
                                               const Rendezvous::Args& recv_args,
                                               const TensorMetaData* meta_data,
                                               Tensor val,
                                               DoneCallback done) {
  string key(std::move(parsed.FullKey().ToString()));
  SetTensorMetaData(key, val.dtype(), val.shape);
  RecvTensorContent(parsed, recv_args, meta_data, done);
}

#endif /*0*/

#if 0

void UcxRemoteRendezvous::UcxRecv(void /*ucp_worker_h ucp_worker*/) {
  ucp_tag_recv_info_t info_tag;
  ucp_tag_message_h msg_tag;
  ucs_status_t status;
  struct msg *msg = 0;
  struct ucx_context *request = 0;
  size_t msg_len = 0;

  printf("UCX receive\n");

  do {
      /* Progressing before probe to update the state */
      ucp_worker_progress(ucp_worker);

      /* Probing incoming events in non-block mode */
      msg_tag = ucp_tag_probe_nb(ucp_worker, 0x1337a880u/*tag*/, -1/*tag_mask*/, 1, &info_tag);
  } while (msg_tag == NULL);

  msg = malloc(info_tag.length);
//  CHKERR_JUMP(!msg, "allocate memory\n", err);
  request = ucp_tag_msg_recv_nb(ucp_worker, msg, info_tag.length,
                                ucp_dt_make_contig(1), msg_tag, NULL/*recv_handle*/);


  if (UCS_PTR_IS_ERR(request)) {
      fprintf(stderr, "unable to receive UCX data message (%u)\n",
              UCS_PTR_STATUS(request));
      free(msg);
      goto err_ep;
  } else {
      wait(ucp_worker, request);
      request->completed = 0;
      ucp_request_release(request);
      printf("UCX data message was received\n");
  }

}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

Status UcxRemoteRendezvous::Send(const Rendezvous::ParsedKey& parsed,
                                 const Rendezvous::Args& args,
                                 const Tensor& val, const bool is_dead) {
  return Status::OK();
}

#if 0

Status UcxRemoteRendezvous::Send(const Rendezvous::ParsedKey& parsed,
                                  const Rendezvous::Args& args,
                                  const Tensor& val, const bool is_dead) {

  VLOG(1) << "UcxRemoteRendezvous Send " << this << " " << parsed.FullKey();
//  CHECK(is_initialized()) << "Send called when uninitialized.";
//  Status s = ValidateDevices(parsed, false /*!is_src*/);
//  if (!s.ok()) {
//  done(s, Args(), recv_args, Tensor(), false);
//  return;
//  }

  // Are src and dst in the same worker?
  if (IsSameWorker(parsed.src, parsed.dst)) {
    mutex_lock l(mu_);
    if (!s.ok()) return s;
    DCHECK(is_initialized_locked());
    if (!IsLocalDevice(session_->worker_name, parsed.src_device)) {
      return errors::InvalidArgument(
          "Invalid rendezvous key (src): ", parsed.FullKey(), " @ ",
          session_->worker_name);
    }
    // Buffers "val" and "device_context" in local_.
    return local_->Send(parsed, args, val, is_dead);
  }
  else {
    SendToRemoteAsync(//Params);
  }
}

void UcxRendezvousMgr::SendToRemoteAsync(const Rendezvous::ParsedKey& parsed,
                       const Rendezvous::Args& args,
                       const Tensor& val
                       /*Params*/) {

  string key(std::move(parsed.FullKey().ToString()));
  const TensorMetaData* meta_data = GetTensorMetaData(key);
  if (meta_data == nullptr)
  {
    TensorSize = val.TotalBytes();
    SendTensorMetaData(val.dtype(), val.shape, val.is_dead);
  }
  else
  {
    SendTensorContentRequest(rc, request, step_id_, parsed, recv_args, meta_data, done);
  }
}

#endif /*0*/

UcxRendezvousMgr::UcxRendezvousMgr(const WorkerEnv* env)
    : BaseRendezvousMgr(env) {}

BaseRemoteRendezvous* UcxRendezvousMgr::Create(int64 step_id,
                                               const WorkerEnv* worker_env) {
  return new UcxRemoteRendezvous(worker_env, step_id, ucx_mgr_);
}

}  // end namespace tensorflow
