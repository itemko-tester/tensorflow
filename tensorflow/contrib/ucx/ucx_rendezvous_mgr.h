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

#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_UCX_UCX_RENDEZVOUS_MGR_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_UCX_UCX_RENDEZVOUS_MGR_H_

#include "tensorflow/core/distributed_runtime/base_rendezvous_mgr.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/contrib/ucx/ucx_mgr.h"
#include <ucp/api/ucp.h>
#include "tensorflow/contrib/ucx/ucx_service.pb.h"

#define UCX_RENDEZVOUS_MGR_META_DATA_SIZE (256)

namespace tensorflow {

class UcxRemoteRendezvous : public BaseRemoteRendezvous {
 public:
  UcxRemoteRendezvous(const WorkerEnv* env, int64 step_id, UcxMgr* ucx_mgr)
      : BaseRemoteRendezvous(env, step_id), ucx_mgr_(ucx_mgr) {}

 protected:
  void RecvFromRemoteAsync(const Rendezvous::ParsedKey& parsed,
                           const Rendezvous::Args& args,
                           DoneCallback done) override;

  void UcxRecv(const Rendezvous::ParsedKey& parsed, ucp_worker_h ucp_worker,
               const Rendezvous::Args& recv_args, Device* dst_dev,
               DoneCallback done);

  // Need to implement:
  void UcxSend(ucp_worker_h ucp_worker, ucp_ep_h ep, uint64_t* msg,
               size_t msg_len, ucp_tag_t tag);

  Status Send(const Rendezvous::ParsedKey& parsed, const Rendezvous::Args& args,
              const Tensor& val, const bool is_dead) override;

 private:
  static void RecvHandler(void* request, ucs_status_t status,
                          ucp_tag_recv_info_t* info);

  static void RecvMetaDataHandler(void* request, ucs_status_t status,
                                  ucp_tag_recv_info_t* info);

  static void SendHandle(void* request, ucs_status_t status);

  void SendContent(const Tensor& val, size_t size);

  class UcxMetaData {
   public:
    UcxMetaData(bool is_dead, DataType dtype, TensorShape tensor_shape,
                size_t proto_size)
        : is_dead_(is_dead),
          dtype_(dtype),
          tensor_shape_(tensor_shape),
          proto_size_(proto_size) {}
    UcxMetaData(const UcxTensorMetaData& proto)
        : is_dead_(proto.is_dead()),
          dtype_(proto.dtype()),
          tensor_shape_(proto.tensor_shape()),
          proto_size_(proto.proto_size()) {}
    bool is_dead_;
    DataType dtype_;
    TensorShape tensor_shape_;
    size_t proto_size_;
  };

  class UcxTensorRequest {
   public:
    UcxTensorRequest(ucp_worker_h ucp_worker,
                     const Rendezvous::ParsedKey& parsed,
                     const Rendezvous::Args& recv_args, int64 step_id,
                     Device* dst_dev, DoneCallback& done)
        : ucp_worker_(ucp_worker),
          parsed_(parsed),
          recv_args_(recv_args),
          step_id_(step_id),
          dst_dev_(dst_dev),
          done_(done),
          data_msg_(nullptr),
          meta_data_(nullptr),
          result_tensor_(nullptr) {}
    ~UcxTensorRequest() {
      if (result_tensor_ != nullptr) {
        delete result_tensor_;
        result_tensor_ = nullptr;
      }
      if (meta_data_ != nullptr) {
        delete meta_data_;
        meta_data_ = nullptr;
      }
    }

    void RecvTensorMetaData();
    void RecvTensorContent();
    void SetMetaData(UcxMetaData* meta_data) { meta_data_ = meta_data; }
    void SetDataMsg(void* data_msg) { data_msg_ = data_msg; }
    void SetResultTensor(Tensor* result_tensor) {
      result_tensor_ = result_tensor;
    }
    char* GetMetaDataMsg() { return meta_data_msg_; }

   private:
    void Done(const Status& s);
    ucp_worker_h ucp_worker_;
    Rendezvous::ParsedKey parsed_;
    Rendezvous::Args recv_args_;
    int64 step_id_;
    Device* dst_dev_;
    DoneCallback done_;
    char meta_data_msg_[UCX_RENDEZVOUS_MGR_META_DATA_SIZE];
    void* data_msg_;
    UcxMetaData* meta_data_;
    Tensor* result_tensor_;
  };

  ~UcxRemoteRendezvous() override {}

 private:
  UcxMgr* ucx_mgr_;

  TF_DISALLOW_COPY_AND_ASSIGN(UcxRemoteRendezvous);
};

class UcxRendezvousMgr : public BaseRendezvousMgr {
 public:
  explicit UcxRendezvousMgr(const WorkerEnv* env);
  void SetUcxMgr(UcxMgr* ucx_mgr) { ucx_mgr_ = ucx_mgr; }

 protected:
  BaseRemoteRendezvous* Create(int64 step_id,
                               const WorkerEnv* worker_env) override;

 private:
  UcxMgr* ucx_mgr_;
  TF_DISALLOW_COPY_AND_ASSIGN(UcxRendezvousMgr);
};
}
#endif /* THIRD_PARTY_TENSORFLOW_CONTRIB_UCX_UCX_RENDEZVOUS_MGR_H_ */
