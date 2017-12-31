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
#include "tensorflow/contrib/ucx/ucx_util.h"
#include "tensorflow/core/framework/tensor.h"
#include <unordered_set>
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
    const Rendezvous::ParsedKey& parsed, const Rendezvous::Args& args,
    DoneCallback done) {
  Status s;
  Device* dst_dev;
  ucp_worker_h ucp_worker = ucx_mgr_->GetWorker();

  s = env_->device_mgr->LookupDevice(parsed.dst_device, &dst_dev);
  if (!s.ok()) {
    done(s, Args(), args, Tensor(), true);
    return;
  }

  UcxRecv(parsed, ucp_worker, args, dst_dev, done);
}

void UcxRemoteRendezvous::UcxTensorRequest::RecvTensorMetaData() {
  void* request;
  int recv_data_size = 0;
  UcxTensorMetaData* tensor_proto =
      reinterpret_cast<UcxTensorMetaData*>(meta_data_msg_);
  meta_data_ = new UcxMetaData(*tensor_proto);
  bool can_memcpy = DataTypeCanUseMemcpy(meta_data_->dtype_);

  string key(std::move(parsed_.FullKey().ToString()));
  ucp_tag_t tag = UcxUtil::CalcTag(key, step_id_, 1);

  if (meta_data_->is_dead_) {
    // TODO: handle in the future
  }
  // String case (can't DMA)
  if (can_memcpy) {
    Tensor* result_tensor =
        new Tensor(dst_dev_->GetAllocator(recv_args_.alloc_attrs),
                   meta_data_->dtype_, meta_data_->tensor_shape_);
    // TODO: in the future handle GPU copy
    data_msg_ = DMAHelper::base(result_tensor);
    recv_data_size = result_tensor->TotalBytes();
  } else {
    data_msg_ = malloc(meta_data_->proto_size_);
    CHECK(data_msg_ != nullptr) << ": allocate memory failed";
    recv_data_size = meta_data_->proto_size_;
  }
  request = ucp_tag_recv_nb(ucp_worker_, data_msg_, recv_data_size,
                            ucp_dt_make_contig(1), tag, -1, &RecvHandler);
  if (UCS_PTR_IS_ERR(request)) {
    VLOG(ERROR) << "unable to receive UCX data message"
                << UCS_PTR_STATUS(request);
    if (!can_memcpy) {
      free(data_msg_);
    }
  }
  ucp_request_release(request);
}

void UcxRemoteRendezvous::UcxTensorRequest::RecvTensorContent() {
  bool can_memcpy = DataTypeCanUseMemcpy(meta_data_->dtype_);
  // String
  if (can_memcpy) {
    // TODO: CPU to GPU copy --> in the future
    Done(Status::OK());
  } else {
    TensorProto proto;
    CHECK(ParseProtoUnlimited(&proto, data_msg_, meta_data_->proto_size_))
        << "fail to parse proto from array";
    free(data_msg_);
    Status s = dst_dev_->MakeTensorFromProto(proto, recv_args_.alloc_attrs,
                                             result_tensor_);
    Done(s);
  }
}

void UcxRemoteRendezvous::UcxTensorRequest::Done(const Status& s) {
  Tensor val = std::move(*result_tensor_);
  Rendezvous::Args recv_args = std::move(recv_args_);
  bool is_dead = meta_data_->is_dead_;
  DoneCallback done = done_;
  // Should be called last:
  delete (this);
  done(s, Rendezvous::Args(), recv_args, val, is_dead);
}

void UcxRemoteRendezvous::RecvHandler(void* request, ucs_status_t status,
                                      ucp_tag_recv_info_t* info) {
  UcxTensorRequest* tensor_request =
      reinterpret_cast<UcxTensorRequest*>(request);
  VLOG(INFO) << "[0x" << (unsigned int)pthread_self() << "]"
             << "RecvHandler called with status" << status
             << ucs_status_string(status) << ",length" << info->length;
  tensor_request->RecvTensorContent();
}

void UcxRemoteRendezvous::RecvMetaDataHandler(void* request_meta_data,
                                              ucs_status_t status,
                                              ucp_tag_recv_info_t* info) {
  UcxTensorRequest* request =
      reinterpret_cast<UcxTensorRequest*>(request_meta_data);
  VLOG(INFO) << "[0x" << (unsigned int)pthread_self() << "]"
             << "RecvMetaDataHandler called with status" << status
             << ucs_status_string(status) << ",length" << info->length;
  request->RecvTensorMetaData();
}

void UcxRemoteRendezvous::UcxRecv(const Rendezvous::ParsedKey& parsed,
                                  ucp_worker_h ucp_worker,
                                  const Rendezvous::Args& recv_args,
                                  Device* dst_dev, DoneCallback done) {
  UcxTensorRequest* context = new UcxTensorRequest(
      ucp_worker, parsed, recv_args, step_id_, dst_dev, done);

  string key(std::move(parsed.FullKey().ToString()));
  ucp_tag_t tag = UcxUtil::CalcTag(key, step_id_, 0);

  context = (UcxTensorRequest*)ucp_tag_recv_nb(
      ucp_worker, context->GetMetaDataMsg(), UCX_RENDEZVOUS_MGR_META_DATA_SIZE,
      ucp_dt_make_contig(1), tag, -1, RecvMetaDataHandler);
  if (UCS_PTR_IS_ERR(context)) {
    VLOG(ERROR) << "unable to receive UCX meta data message"
                << UCS_PTR_STATUS(context);
    delete (context);
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

void UcxRemoteRendezvous::SendHandle(void* request, ucs_status_t status) {
  // struct ucx_context* context = (struct ucx_context*)request;
  // context->completed = 1;

  VLOG(INFO) << "[0x" << (unsigned int)pthread_self() << "]"
             << "send handler called with status" << status
             << ucs_status_string(status);
}

void UcxRemoteRendezvous::UcxSend(ucp_worker_h ucp_worker, ucp_ep_h ep,
                                  uint64_t* msg, size_t msg_len,
                                  ucp_tag_t tag) {
  struct ucx_context* request = 0;

  request = (struct ucx_context*)ucp_tag_send_nb(
      ep, msg, msg_len, ucp_dt_make_contig(1), tag, &SendHandle);
  if (UCS_PTR_IS_ERR(request)) {
    VLOG(ERROR) << "unable to send UCX data message" << UCS_PTR_STATUS(request);
    free(msg);
  } else if (UCS_PTR_STATUS(request) != UCS_OK) {
    VLOG(INFO) << "UCX data message was scheduled for send\n";
    // WaitForComplete(ucp_worker, request);
    // request->completed = 0;
    ucp_request_release(request);
  }
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

}  // end namespace tensorflow
