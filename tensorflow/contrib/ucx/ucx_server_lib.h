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

#ifndef TENSORFLOW_CONTRIB_UCX_UCX_SERVER_LIB_H_
#define TENSORFLOW_CONTRIB_UCX_UCX_SERVER_LIB_H_

#ifndef TENSORFLOW_USE_UCX

#include <memory>

#include "tensorflow/contrib/ucx/grpc_ucx_service.h"
#include "tensorflow/contrib/ucx/ucx_rendezvous_mgr.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_server_lib.h"

namespace tensorflow {

class UCXServer : public GrpcServer {
 protected:
  UCXServer(const ServerDef& server_def, Env* env);

 public:
  static Status Create(const ServerDef& server_def, Env* env,
                       std::unique_ptr<ServerInterface>* out_server);

  // Destruction is only supported in the factory method. Clean
  // shutdown is not currently implemented for this server type.
  ~UCXServer() override;

  // Implementations of ServerInterface methods.
  Status Start() override;
  Status Join() override;

 protected:
  Status Init(ServiceInitFunction service_func,
              RendezvousMgrCreationFunction rendezvous_mgr_func);
  Status ChannelCacheFactory(const ServerDef& server_def,
                             GrpcChannelCache** channel_cache);

 private:
  // Guards state transitions.
  mutex mu_;

  enum State {
    DISCONNECTED,
    CONNECTED
  };
  State ucx_state_ GUARDED_BY(mu_);
  UcxMgr* ucx_mgr_;
  GrpcUcxService* ucx_service_ = nullptr;
  std::unique_ptr<Thread> ucx_thread_ GUARDED_BY(mu_);
  GrpcChannelCache* channel_cache_ = nullptr;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_USE_UCX
#endif  // TENSORFLOW_CONTRIB_UCX_UCX_SERVER_LIB_H_
