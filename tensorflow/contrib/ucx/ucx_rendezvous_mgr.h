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

#include <ucp/api/ucp.h>
#include "tensorflow/contrib/ucx/ucx_mgr.h"
#include "tensorflow/contrib/ucx/ucx_service.pb.h"
#include "tensorflow/core/distributed_runtime/base_rendezvous_mgr.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/macros.h"

#define UCX_RENDEZVOUS_MGR_META_DATA_SIZE (256)

namespace tensorflow {

class UcxRemoteRendezvous : public BaseRemoteRendezvous {
 public:
  UcxRemoteRendezvous(const WorkerEnv* env, int64 step_id, UcxMgr* ucx_mgr)
      : BaseRemoteRendezvous(env, step_id), ucx_mgr_(ucx_mgr) {}

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

    std::ostream& print(std::ostream& out) const {
      out << "Dtype = " << DataTypeString(dtype_)
          << ", Shape = " << tensor_shape_.DebugString() << ", Proto size = 0x"
          << std::hex << proto_size_ << ", Is dead = " << is_dead_;
      return out;
    }
  };

  class UcxTensorRecv {
   public:
    UcxTensorRecv(ucp_worker_h ucp_worker, const Rendezvous::ParsedKey& parsed,
                  const Rendezvous::Args& recv_args, int64 step_id,
                  Device* dst_dev, DoneCallback& done)
        : ucp_worker_(ucp_worker),
          key_((parsed.FullKey().ToString())),
          recv_args_(recv_args),
          step_id_(step_id),
          dst_dev_(dst_dev),
          done_(done),
          data_size_(0),
          data_msg_(nullptr),
          meta_data_(nullptr),
          result_tensor_(nullptr) {
      memset(meta_data_msg_, 0, UCX_RENDEZVOUS_MGR_META_DATA_SIZE);
    }
    ~UcxTensorRecv() {
      if (result_tensor_ != nullptr) {
        delete result_tensor_;
        result_tensor_ = nullptr;
      }
      if (meta_data_ != nullptr) {
        delete meta_data_;
        meta_data_ = nullptr;
      }
    }

    void Start(mutex& mtx);
    void RecvTensorMetaData();
    void RecvTensorContent();

    struct ContextWrap {
      UcxRemoteRendezvous::UcxTensorRecv* context;
      size_t len;
    };

   private:
    static void WaitForContext(void* request, ucs_status_t status,
                               ucp_tag_recv_info_t* info, string func_name,
                               ContextWrap* ctx);
    static void RecvTensorContentHandler(void* request, ucs_status_t status,
                                         ucp_tag_recv_info_t* info);
    static void RecvMetaDataHandler(void* request, ucs_status_t status,
                                    ucp_tag_recv_info_t* info);
    void Done(const Status& s);
    ucp_worker_h ucp_worker_;
    string key_;
    Rendezvous::Args recv_args_;
    int64 step_id_;
    Device* dst_dev_;
    DoneCallback done_;
    char meta_data_msg_[UCX_RENDEZVOUS_MGR_META_DATA_SIZE];
    size_t data_size_;
    void* data_msg_;
    UcxMetaData* meta_data_;
    Tensor* result_tensor_;
  };

 protected:
  void RecvFromRemoteAsync(const Rendezvous::ParsedKey& parsed,
                           const Rendezvous::Args& args,
                           DoneCallback done) override;

  Status Send(const Rendezvous::ParsedKey& parsed, const Rendezvous::Args& args,
              const Tensor& val, const bool is_dead) override;

 private:
  class UcxTensorSend {
   public:
    UcxTensorSend(ucp_ep_h ep, const Rendezvous::ParsedKey& parsed,
                  const Rendezvous::Args& send_args, int64 step_id,
                  const Tensor& val, bool is_dead)
        : ep_(ep),
          key_((parsed.FullKey().ToString())),
          send_args_(send_args),
          step_id_(step_id),
          val_(val),
          data_msg_(nullptr),
          is_dead_(is_dead),
          tensor_buffer_(nullptr),
          is_meta_data_send_(false) {}

    ~UcxTensorSend() {}

    void Start(mutex& mtx);
    void SendTensorMetaData(mutex& mtx);
    void SendTensorContent(mutex& mtx);

   private:
    void SendDone();
    static UcxTensorSend* WaitForContext(void* request, ucs_status_t status,
                                         string func_name);
    static void SendMetaDataHandler(void* request, ucs_status_t status);
    static void SendTensorContentHandler(void* request, ucs_status_t status);
    ucp_ep_h ep_;
    string key_;
    Rendezvous::Args send_args_;
    int64 step_id_;
    Tensor val_;
    char meta_data_msg_[UCX_RENDEZVOUS_MGR_META_DATA_SIZE];
    void* data_msg_;
    bool is_dead_;
    TensorBuffer* tensor_buffer_;
    bool is_meta_data_send_;
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
