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

#include "tensorflow/contrib/ucx/ucx_mgr.h"
#include "tensorflow/core/distributed_runtime/base_rendezvous_mgr.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/platform/macros.h"

#include "tensorflow/contrib/ucx/ucx_service.pb.h"

namespace tensorflow {

class TensorMetaData {
 public:
  TensorMetaData(DataType data_type, TensorShape tensor_shape)
      : data_type_(data_type), tensor_shape_(tensor_shape) {}

  DataType data_type_;
  TensorShape tensor_shape_;
};

class UcxRemoteRendezvous : public BaseRemoteRendezvous {
 public:
  UcxRemoteRendezvous(const WorkerEnv* env, int64 step_id, UcxMgr* ucx_mgr)
      : BaseRemoteRendezvous(env, step_id), ucx_mgr_(ucx_mgr) {}

 protected:
  void RecvFromRemoteAsync(const Rendezvous::ParsedKey& parsed,
                           const Rendezvous::Args& args,
                           DoneCallback done) override;

  void RecvMetaDataCallback(const Rendezvous::ParsedKey& parsed,
                            const Rendezvous::Args& args, DoneCallback done);

  void RecvContentCallback(const Rendezvous::ParsedKey& parsed,
                           const Rendezvous::Args& args, DoneCallback done);

  void RecvTensorContent(const Rendezvous::ParsedKey& parsed,
                         const Rendezvous::Args& recv_args,
                         const TensorMetaData* meta_data, DoneCallback done);

  void RecvTensorMetaData(/*TODO params*/);

  // void UcxRecv(void /*ucp_worker_h ucp_worker*/);

  Status Send(const Rendezvous::ParsedKey& parsed, const Rendezvous::Args& args,
              const Tensor& val, const bool is_dead) override;

  //  const TensorMetaData* GetTensorMetaData(const std::string& tensor_name) {
  //    mutex_lock l(tensor_sizes_mu_);
  //    auto it = tensors_meta_data_.find(tensor_name);
  //    if (it == tensors_meta_data_.end()) {
  //      return nullptr;
  //    }
  //    return &it->second;
  //  }

  //  // Return true if inserted new
  //  TensorMetaData* SetTensorMetaData(const std::string& tensor_name, DataType
  //  dtype, const TensorShape& shape) {
  //    mutex_lock l(tensor_sizes_mu_);
  //    TensorMetaData meta_data(dtype, shape);
  //    auto res = tensors_meta_data_.insert(std::make_pair(tensor_name,
  //    meta_data));
  //    return &res.first->second;
  //  }
  //
 private:
  //  void SendToRemoteAsync(const Rendezvous::ParsedKey& parsed,
  //                         const Rendezvous::Args& args,
  //                         /*Params*/);
  //
  //  void SendMetaData(/*params*/);

  void SendContent(const Tensor& val, size_t size);

  ~UcxRemoteRendezvous() override {}
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
