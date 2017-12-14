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

#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_UCX_UCX_MGR_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_UCX_UCX_MGR_H_

#ifndef TENSORFLOW_USE_UCX

#include <unordered_map>

#include "tensorflow/core/distributed_runtime/rpc/grpc_channel.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/contrib/ucx/ucx_channel.h"

namespace tensorflow {

class UcxMgr {

 public:
  explicit UcxMgr(const WorkerEnv* const worker_env,
                  GrpcChannelCache* const channel_cache);
  ~UcxMgr();

 private:
  string local_worker_;
  size_t num_remote_workers_;
  ucp_context_h ucp_context_;
  ucp_worker_h ucp_worker_;
  ucp_address_t* local_addr_;
  size_t local_addr_len_;
  const WorkerEnv* const worker_env_;
  typedef std::unordered_map<string, UcxChannel*> ChannelTable;
  ChannelTable channel_table_;
  TF_DISALLOW_COPY_AND_ASSIGN(UcxMgr);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_USE_UCX
#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_UCX_UCX_MGR_H_
